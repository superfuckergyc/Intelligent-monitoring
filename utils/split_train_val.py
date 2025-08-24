import shutil
from pathlib import Path
import random
import argparse

def split_images(img_root, val_ratio=0.2, seed=42):
    random.seed(seed)
    img_root = Path(img_root)
    train_dir = img_root / "train"
    val_dir = img_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片（常见后缀）
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(img_root.glob(ext))

    imgs = [p for p in imgs if p.parent == img_root]  # 只取根下的，不包含已存在train/val里的
    random.shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    val_set = set(imgs[:n_val])

    for p in imgs:
        dst = val_dir / p.name if p in val_set else train_dir / p.name
        shutil.move(str(p), str(dst))

    print(f"[OK] Split {len(imgs)} images → train:{len(imgs)-n_val}, val:{n_val}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="datasets/shiprs/images")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()
    split_images(args.images, args.val_ratio)
