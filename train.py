import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from tqdm import tqdm

from src.dataset import Sentinel
from src.pix2pix import Pix2Pix
from src.plot import TrainingPlotter
from utils.config import Config
from utils.utils import setup_logging


def save_checkpoint(model: Pix2Pix, epoch: int, checkpoint_dir: Path, config: Config):
    """Save model checkpoint safely inside the unique run folder"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    gen_filename = f"generator_epoch_{epoch}.pth"
    gen_path = checkpoint_dir / gen_filename

    disc_filename = f"discriminator_epoch_{epoch}.pth"
    disc_path = checkpoint_dir / disc_filename

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.save_model(str(gen_path), str(disc_path))
    config.save(checkpoint_dir / "config.yaml")


def load_checkpoint(model: Pix2Pix, config: Config, kw_args=defaultdict(lambda: None)):
    """Load model checkpoint for continuing execution runs"""
    gen_checkpoint = Path(kw_args["gen_checkpoint"] or config["training"]["gen_checkpoint"])
    disc_checkpoint = Path(kw_args["disc_checkpoint"] or config["training"]["disc_checkpoint"])

    if not gen_checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint file not found: {gen_checkpoint}")
    if not disc_checkpoint.exists():
        raise FileNotFoundError(f"Discriminator checkpoint file not found: {disc_checkpoint}")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_model(gen_path=str(gen_checkpoint), disc_path=str(disc_checkpoint))


def create_dataloader(config, split_type: str, input_transform, kw_args=defaultdict(lambda: None)):
    """Create data loading utility instances"""
    dataset = Sentinel(
        root_dir=kw_args["root_dir"] or config["dataset"]["root_dir"],
        split_type=split_type,
        input_transform=input_transform,
        target_transform=input_transform,
        split_mode=config["dataset"]["split_mode"],
        split_ratio=config["dataset"]["split_ratio"],
        seed=config["dataset"]["seed"],
    )
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["dataset"]["shuffle"] if split_type == "train" else False,
        num_workers=config["training"]["num_workers"],
    )


def train_epoch(model, train_loader, device, epoch, tb_writer):
    """Train for one epoch and push performance indicators to TensorBoard"""
    model.train()
    total_lossD, total_lossG = 0.0, 0.0
    total_lossG_GAN, total_lossG_L1 = 0.0, 0.0

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    with tqdm(train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
        for real_images, target_images in pbar:
            real_images, target_images = real_images.to(device), target_images.to(device)
            losses = raw_model.train_step(real_images, target_images)
            
            total_lossD += losses["loss_D"]
            total_lossG += losses["loss_G"]
            total_lossG_GAN += losses["loss_G_GAN"]
            total_lossG_L1 += losses["loss_G_L1"]
            
            pbar.set_postfix({"loss_D": losses["loss_D"], "loss_G": losses["loss_G"]})

    num_steps = len(train_loader)
    metrics = {
        "Train/Loss_D": total_lossD / num_steps,
        "Train/Loss_G": total_lossG / num_steps,
        "Train/Loss_G_GAN": total_lossG_GAN / num_steps,
        "Train/Loss_G_L1": total_lossG_L1 / num_steps,
    }

    if tb_writer:
        for name, value in metrics.items():
            tb_writer.add_scalar(name, value, epoch)

    return metrics["Train/Loss_D"], metrics["Train/Loss_G"], metrics["Train/Loss_G_GAN"], metrics["Train/Loss_G_L1"]


def validate(model, val_loader, device, epoch, tb_writer):
    """Validate model and write isolated validation parameters to TensorBoard"""
    model.eval()
    total_lossD, total_lossG = 0.0, 0.0
    total_lossG_GAN, total_lossG_L1 = 0.0, 0.0

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    with torch.no_grad():
        for real_images, target_images in val_loader:
            real_images, target_images = real_images.to(device), target_images.to(device)
            losses = raw_model.validation_step(real_images, target_images)
            
            total_lossD += losses["loss_D"]
            total_lossG += losses["loss_G"]
            total_lossG_GAN += losses["loss_G_GAN"]
            total_lossG_L1 += losses["loss_G_L1"]

    num_steps = len(val_loader)
    metrics = {
        "Val/Loss_D": total_lossD / num_steps,
        "Val/Loss_G": total_lossG / num_steps,
        "Val/Loss_G_GAN": total_lossG_GAN / num_steps,
        "Val/Loss_G_L1": total_lossG_L1 / num_steps,
    }

    if tb_writer:
        for name, value in metrics.items():
            tb_writer.add_scalar(name, value, epoch)

    return metrics["Val/Loss_D"], metrics["Val/Loss_G"], metrics["Val/Loss_G_GAN"], metrics["Val/Loss_G_L1"]


def train(kw_args=defaultdict(lambda: None)):
    config = Config("config.yaml")
    use_validation = kw_args["use_validation"] or config["training"]["use_validation"]

    # Generate isolated directories for this unique execution run
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config["logging"]["local"]["base_dir"]) / run_id
    checkpoint_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"

    for folder in [checkpoint_dir, plots_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    setup_logging(config)

    # Instantiate TensorBoard SummaryWriter
    tb_writer = None
    if config["logging"]["tensorboard"]["enabled"]:
        tb_writer = SummaryWriter(log_dir=str(Path(config["logging"]["tensorboard"]["log_dir"]) / run_id))

    device = torch.device(kw_args["device"] or config["training"]["device"])

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    train_loader = create_dataloader(config, "train", train_transforms, kw_args=kw_args)
    val_loader = create_dataloader(config, "val", train_transforms, kw_args=kw_args) if use_validation else None

    model = Pix2Pix(
        c_in=config["model"]["c_in"],
        c_out=config["model"]["c_out"],
        netD=config["model"]["netD"],
        lambda_L1=config["model"]["lambda_L1"],
        is_CGAN=config["model"]["is_CGAN"],
        use_upsampling=config["model"]["use_upsampling"],
        mode=config["model"]["mode"],
        c_hid=config["model"]["c_hid"],
        n_layers=config["model"]["n_layers"],
        lr=config["training"]["lr"],
        beta1=config["training"]["beta1"],
        beta2=config["training"]["beta2"],
    ).to(device)

    start_epoch, end_epoch = 1, config["training"]["num_epochs"] + 1
    if kw_args["resume"] or config["training"]["resume"]:
        load_checkpoint(model, config, kw_args=kw_args)
        start_epoch = kw_args["resume_epoch"] or config["training"].get("resume_epoch", 1)

    model = torch.compile(model)

    G_losses, D_losses, GAN_losses, L1_losses = [], [], [], []
    plotter = TrainingPlotter(plots_dir)

    for epoch in range(start_epoch, end_epoch):
        # Always compute the training iteration parameters
        epoch_D_loss, epoch_G_loss, epoch_GAN_loss, epoch_L1_loss = train_epoch(
            model, train_loader, device, epoch, tb_writer
        )

        if use_validation and val_loader:
            epoch_D_loss, epoch_G_loss, epoch_GAN_loss, epoch_L1_loss = validate(
                model, val_loader, device, epoch, tb_writer
            )
        
        G_losses.append(epoch_G_loss)
        D_losses.append(epoch_D_loss)
        GAN_losses.append(epoch_GAN_loss)
        L1_losses.append(epoch_L1_loss)

        if epoch % config["training"]["save_freq"] == 0:
            save_checkpoint(model, epoch, checkpoint_dir, config)
            plotter.plot_curve(G_losses, "Generator Loss", "Loss", "generator_loss.png")
            plotter.plot_curve(D_losses, "Discriminator Loss", "Loss", "discriminator_loss.png")
            plotter.plot_curve(GAN_losses, "GAN Loss", "Loss", "gan_loss.png")
            plotter.plot_curve(L1_losses, "L1 Loss", "Loss", "l1_loss.png")
            plotter.plot_multiple([G_losses, D_losses], ["Generator", "Discriminator"], "Generator vs Discriminator Loss", "g_d_loss.png")

        history = pd.DataFrame({
            "epoch": range(1, len(G_losses) + 1),
            "generator_loss": G_losses,
            "discriminator_loss": D_losses,
            "gan_loss": GAN_losses,
            "l1_loss": L1_losses
        })
        history.to_csv(logs_dir / "training_history.csv", index=False)

    save_checkpoint(model, config["training"]["num_epochs"], checkpoint_dir, config)

    if tb_writer:
        tb_writer.close()
    print(f"Training successfully complete. Items stored inside directory -> {run_dir}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Train the Pix2Pix model.")
    cli_parser.add_argument("--root_dir", type=str, default=None)
    cli_parser.add_argument("--gen_checkpoint", type=str, default=None)
    cli_parser.add_argument("--disc_checkpoint", type=str, default=None)
    cli_parser.add_argument("--use_validation", action="store_true")
    cli_parser.add_argument("--device", type=str, default=None)
    cli_parser.add_argument("--resume", action="store_true")
    cli_parser.add_argument("--resume_epoch", type=int, default=0)
    cli_args = cli_parser.parse_args()
    
    train(kw_args=defaultdict(lambda: None, vars(cli_args)))