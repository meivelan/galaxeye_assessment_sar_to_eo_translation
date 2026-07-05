"""
Evaluation Script
"""

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2

from src.dataset import Sentinel
from src.metric import calculate_fid, extract_features
from src.pix2pix import Pix2Pix
from utils.config import Config


def evaluate(kw_args=defaultdict(lambda: None)):
    config = Config("config.yaml")
    
    device_str = kw_args['device'] or config["training"]["device"]
    device = torch.device(device_str)

    inception = (
        models.inception_v3(weights="DEFAULT", transform_input=False).eval().to(device)
    )

    inception_transform = v2.Compose(
        [
            v2.Resize(342),
            v2.CenterCrop(299),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = Sentinel(
        root_dir=kw_args['root_dir'] or config["dataset"]["root_dir"],
        split_type="test",
        input_transform=base_transform,
        target_transform=base_transform,
        split_mode=config["dataset"]["split_mode"],
        split_ratio=config["dataset"]["split_ratio"],
        seed=config["dataset"]["seed"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = (
        Pix2Pix(
            c_in=config["model"]["c_in"],
            c_out=config["model"]["c_out"],
            netD=config["model"].get("netD"),
            is_CGAN=config["model"].get("is_CGAN", True),
            use_upsampling=config["model"]["use_upsampling"],
            mode=config["model"]["mode"],
        )
        .to(device)
        .eval()
    )

    gen_checkpoint = Path(kw_args['gen_checkpoint'] or config["training"]["gen_checkpoint"])

    if not gen_checkpoint.exists():
        raise FileNotFoundError(
            f"Generator checkpoint file not found: {gen_checkpoint}\nPlease check config.yaml"
        )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_model(gen_path=str(gen_checkpoint))

    target_features = []
    fake_features = []

    with torch.no_grad():
        for real_images, target_images in dataloader:
            real_images, target_images = real_images.to(device), target_images.to(device)

            # Generate fake image arrays
            fake_images = raw_model.generate(real_images, is_scaled=True, to_uint8=True)

            # Map validation references to [0, 255] uint8 arrays to line up target metrics
            target_images_uint8 = (target_images * 255).to(dtype=torch.uint8)
            
            target_images_preprocessed = inception_transform(target_images_uint8)
            target_feats = extract_features(target_images_preprocessed, inception)
            target_features.append(target_feats.cpu().numpy())

            fake_images_preprocessed = inception_transform(fake_images)
            fake_feats = extract_features(fake_images_preprocessed, inception)
            fake_features.append(fake_feats.cpu().numpy())

    real_features = np.concatenate(target_features, axis=0)
    generated_features = np.concatenate(fake_features, axis=0)

    fid_score = calculate_fid(real_features, generated_features)
    print(f"Evaluation Complete | FID Score: {fid_score:.4f}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Evaluation Script")
    cli_parser.add_argument("--root_dir", type=str, default=None)
    cli_parser.add_argument("--device", type=str, default=None)
    cli_parser.add_argument("--gen_checkpoint", type=str, default=None)
    cli_args = cli_parser.parse_args()
    
    evaluate(kw_args=defaultdict(lambda: None, vars(cli_args)))