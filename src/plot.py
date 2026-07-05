import os
from pathlib import Path
import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_curve(self, values, title, ylabel, filename):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(values) + 1), values, linewidth=2, marker='o', markersize=3)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(self.save_dir / filename, dpi=300)
        plt.close()

    def plot_multiple(self, curves, labels, title, filename):
        plt.figure(figsize=(8, 5))

        for curve, label in zip(curves, labels):
            plt.plot(range(1, len(curve) + 1), curve, label=label)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.save_dir / filename, dpi=300)
        plt.close()