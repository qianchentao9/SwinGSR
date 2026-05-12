"""Download DIV2K (and Set5 for validation) from HuggingFace into the layout
that SwinGSR's yaml configs expect:

    datasets/DIV2K/DIV2K_train_HR/0001.png ...
    datasets/DIV2K/DIV2K_train_LR_bicubic/X2/0001x2.png ...
    datasets/DIV2K/DIV2K_train_LR_bicubic/X4/0001x4.png ...
    datasets/Set5/GTmod12/*.png
    datasets/Set5/LRbicx4/*.png
"""
import shutil
from pathlib import Path
from datasets import load_dataset

ROOT = Path("datasets")


def copy_split(ds, hr_dir: Path, lr_dir: Path):
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    for ex in ds:
        shutil.copy(ex["hr"], hr_dir / Path(ex["hr"]).name)
        shutil.copy(ex["lr"], lr_dir / Path(ex["lr"]).name)


print(">>> [1/3] DIV2K train, bicubic x4 ...")
ds_x4 = load_dataset("eugenesiow/Div2k", "bicubic_x4", split="train")
copy_split(
    ds_x4,
    ROOT / "DIV2K/DIV2K_train_HR",
    ROOT / "DIV2K/DIV2K_train_LR_bicubic/X4",
)

print(">>> [2/3] DIV2K train, bicubic x2 (only LR, HR already copied) ...")
ds_x2 = load_dataset("eugenesiow/Div2k", "bicubic_x2", split="train")
lr_x2 = ROOT / "DIV2K/DIV2K_train_LR_bicubic/X2"
lr_x2.mkdir(parents=True, exist_ok=True)
for ex in ds_x2:
    shutil.copy(ex["lr"], lr_x2 / Path(ex["lr"]).name)

print(">>> [3/3] Set5 validation, bicubic x4 ...")
val = load_dataset("eugenesiow/Set5", "bicubic_x4", split="validation")
copy_split(
    val,
    ROOT / "Set5/GTmod12",
    ROOT / "Set5/LRbicx4",
)

# Set5 x2 for the x2 training config
print(">>> Set5 validation, bicubic x2 ...")
val2 = load_dataset("eugenesiow/Set5", "bicubic_x2", split="validation")
lr_bic_x2 = ROOT / "Set5/LRbicx2"
lr_bic_x2.mkdir(parents=True, exist_ok=True)
for ex in val2:
    shutil.copy(ex["lr"], lr_bic_x2 / Path(ex["lr"]).name)

print("Done.")
