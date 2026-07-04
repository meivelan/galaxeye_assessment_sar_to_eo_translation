import json
import shutil
from pathlib import Path
import os

def split_dataset():
    # Load split file
    with open("datasets/split.json", "r") as f:
        split_file = json.load(f)

    old_data_root = Path(
        "/kaggle/input/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/v_2"
    )
    new_data_root = Path("/kaggle/working/sentinel12-image-pairs-segregated-by-terrain")

    # Create output directories
    for sensor in ["s1", "s2"]:
        for split in ["train", "val", "test"]:
            (new_data_root / sensor / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for terrain_dir in old_data_root.iterdir():
        if not terrain_dir.is_dir():
            continue

        for split, files in split_file["data"].items():
            for file in files:
                # S1
                src = terrain_dir / "s1" / file
                dst = new_data_root / "s1" / split / file
                if src.exists():
                    shutil.copy2(src, dst)

                # S2
                s2_file = file.replace("s1", "s2")
                src = terrain_dir / "s2" / s2_file
                dst = new_data_root / "s2" / split / file
                if src.exists():
                    shutil.copy2(src, dst)

    print(f"Created new dataset structure with train, val, and test splits.")
    print(f"Train samples: {len(os.listdir(new_data_root / 's1' / 'train'))}")
    print(f"Val samples: {len(os.listdir(new_data_root / 's1' / 'val'))}")
    print(f"Test samples: {len(os.listdir(new_data_root / 's1' / 'test'))}")