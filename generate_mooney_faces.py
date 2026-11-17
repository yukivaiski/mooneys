#!/usr/bin/env python3
"""
Download (optional) and generate strictly two-tone (0/255) Mooney animal faces
(Cat vs. Dog) from the Oxford-IIIT Pet Dataset with a breed-stratified
train / val / test split.
"""

import argparse
import base64
import csv
import json
import os
import random
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/zippyz/cats-and-dogs-breeds-classification-oxford-dataset"
)


def _get_kaggle_auth_header() -> dict[str, str]:
    """Builds Basic Auth header if KAGGLE_USERNAME/KEY env vars or ~/.kaggle/kaggle.json exist."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not (user and key):
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                creds = json.loads(kaggle_json.read_text(encoding="utf-8"))
                user = creds.get("username")
                key = creds.get("key")
            except Exception:
                pass

    if user and key:
        token = base64.b64encode(f"{user}:{key}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}"}
    return {}


def download_and_extract_dataset(
    target_dir: Path,
    url: str = KAGGLE_DATASET_URL,
    force: bool = False,
) -> Path:
    """Downloads and extracts the Oxford-IIIT Pet dataset zip from Kaggle."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "cats-and-dogs-breeds-classification-oxford-dataset.zip"

    # Check if already extracted
    if not force:
        try:
            discover_dataset_paths(target_dir)
            print(f"[INFO] Dataset already extracted at: {target_dir.resolve()}")
            return target_dir
        except FileNotFoundError:
            pass

    if force or not zip_path.exists():
        print(f"[INFO] Downloading dataset from Kaggle to {zip_path}...")
        headers = {
            "User-Agent": "Mozilla/5.0",
            **_get_kaggle_auth_header(),
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_f:
            total_size = resp.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size and total_size.isdigit() else None
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    print(
                        f"\r  Downloaded {downloaded / (1024**2):.1f} / "
                        f"{total_bytes / (1024**2):.1f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  Downloaded {downloaded / (1024**2):.1f} MB...",
                        end="",
                        flush=True,
                    )
        print("\n[INFO] Download complete.")
    else:
        print(f"[INFO] Using existing zip archive: {zip_path}")

    print(f"[INFO] Extracting archive into {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    print("[INFO] Extraction complete.")

    return target_dir


def load_species_map(ann_dir: Path) -> dict[str, str]:
    """Loads ground-truth species ('cat' or 'dog') from annotations/list.txt if available."""
    species_map = {}
    list_file = ann_dir / "list.txt"
    if not list_file.exists():
        return species_map

    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                stem, _, species_id = parts[0], parts[1], parts[2]
                if species_id == "1":
                    species_map[stem] = "cat"
                elif species_id == "2":
                    species_map[stem] = "dog"
    return species_map


def get_species(stem: str, species_map: dict[str, str]) -> str:
    """Returns 'cat' or 'dog' from list.txt, falling back to Oxford filename casing."""
    if stem in species_map:
        return species_map[stem]
    return "cat" if stem[0].isupper() else "dog"


def parse_head_bbox(xml_path: Path) -> tuple[int, int, int, int]:
    """Extracts (xmin, ymin, xmax, ymax) from a Pascal VOC XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bndbox = root.find(".//bndbox")
    if bndbox is None:
        raise ValueError(f"No <bndbox> found in {xml_path}")

    return (
        int(float(bndbox.find("xmin").text)),
        int(float(bndbox.find("ymin").text)),
        int(float(bndbox.find("xmax").text)),
        int(float(bndbox.find("ymax").text)),
    )


def pad_and_clamp_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    """Adds proportional padding around the head box and clamps to image bounds."""
    xmin, ymin, xmax, ymax = bbox
    pad_x = int((xmax - xmin) * pad_ratio)
    pad_y = int((ymax - ymin) * pad_ratio)

    return (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(img_w, xmax + pad_x),
        min(img_h, ymax + pad_y),
    )


def process_single_face(
    img_path: Path,
    xml_path: Path,
    trimap_path: Path,
    out_path: Path,
    target_size: int = 224,
    blur_strength: int = 15,
    white_ratio: float = 0.50,
    pad_ratio: float = 0.10,
    bg_mode: str = "keep",
    rng: random.Random | None = None,
) -> tuple[bool, str]:
    """Crops head ROI, resizes, thresholds via foreground trimap, and saves binary (0/255) Mooney."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, f"Failed to read image: {img_path}"

    trimap = cv2.imread(str(trimap_path), cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        return False, f"Failed to read trimap: {trimap_path}"

    if img.shape != trimap.shape:
        trimap = cv2.resize(trimap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    try:
        raw_bbox = parse_head_bbox(xml_path)
    except Exception as exc:
        return False, str(exc)

    h, w = img.shape
    xmin, ymin, xmax, ymax = pad_and_clamp_bbox(raw_bbox, w, h, pad_ratio=pad_ratio)
    if xmax <= xmin or ymax <= ymin:
        return False, f"Invalid bounding box in {xml_path}"

    # 1. Crop both image and trimap to head ROI
    face_crop = img[ymin:ymax, xmin:xmax]
    trimap_crop = trimap[ymin:ymax, xmin:xmax]

    # 2. Resize to canonical square resolution so blur_strength is scale-invariant
    face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    trimap_resized = cv2.resize(trimap_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    # 3. Apply Gaussian blur (force odd positive kernel)
    ksize = max(1, int(blur_strength) | 1)
    blurred = cv2.GaussianBlur(face_resized, (ksize, ksize), 0)

    # 4. Calculate threshold strictly on foreground (1) and boundary (3) pixels
    fg_mask = trimap_resized != 2
    fg_pixels = blurred[fg_mask]

    if fg_pixels.size < 25:
        return False, f"Insufficient foreground pixels in {img_path.name}"

    percentile = (1.0 - np.clip(white_ratio, 0.0, 1.0)) * 100.0
    threshold_val = np.percentile(fg_pixels, percentile)

    # Strictly binary {0, 255}
    _, mooney = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    # 5. Strictly binary background options
    if bg_mode == "black":
        mooney[~fg_mask] = 0
    elif bg_mode == "white":
        mooney[~fg_mask] = 255
    elif bg_mode == "alt":
        bg_val = rng.choice([0, 255]) if rng else 0
        mooney[~fg_mask] = bg_val

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mooney)
    return True, str(out_path)


def discover_dataset_paths(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    """Locates images, annotations root, xmls, and trimaps directories."""
    img_candidates = [
        dataset_root / "images" / "images",
        dataset_root / "images",
    ]
    ann_candidates = [
        dataset_root / "annotations" / "annotations",
        dataset_root / "annotations",
    ]

    img_dir = next((p for p in img_candidates if p.is_dir() and any(p.glob("*.jpg"))), None)
    ann_dir = next((p for p in ann_candidates if (p / "xmls").is_dir() and (p / "trimaps").is_dir()), None)

    if img_dir is None or ann_dir is None:
        raise FileNotFoundError(
            f"Could not locate 'images/' and 'annotations/{{xmls,trimaps}}' under {dataset_root}"
        )

    return img_dir, ann_dir, ann_dir / "xmls", ann_dir / "trimaps"


def main():
    parser = argparse.ArgumentParser(
        description="Download & generate strictly binary (0/255) Mooney animal faces with train/val/test splits."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./raw_oxford_dataset"),
        help="Path to the extracted (or target download) Oxford-IIIT Pet dataset directory (default: ./raw_oxford_dataset).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the dataset from Kaggle if not already present in --dataset-dir.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading and re-extracting the Kaggle zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./oxford_mooney_cat_dog"),
        help="Destination directory for train/, val/, and test/ folders (default: ./oxford_mooney_cat_dog).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target square resolution (pixels) before blurring (default: 224).",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=15,
        help="Gaussian blur kernel size applied at target resolution (default: 15).",
    )
    parser.add_argument(
        "--white-ratio",
        type=float,
        default=0.50,
        help="Target ratio of white pixels inside the foreground face mask [0.0 - 1.0] (default: 0.50).",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.10,
        help="Proportional padding added around the XML head bounding box (default: 0.10).",
    )
    parser.add_argument(
        "--bg-mode",
        choices=["keep", "black", "white", "alt"],
        default="keep",
        help="Strictly binary background mode: keep (thresholded), black (0), white (255), or alt (random 0/255).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.70,
        help="Fraction of each breed allocated to train/ (default: 0.70).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of each breed allocated to val/ (remainder goes to test/) (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val/test splits (default: 42).",
    )
    parser.add_argument(
        "--save-paired-rgb",
        action="store_true",
        help="Also save aligned RGB head crops in {train,val,test}_rgb/{cat,dog} for control baselines.",
    )

    args = parser.parse_args()

    if args.train_split + args.val_split >= 1.0 or args.train_split <= 0 or args.val_split <= 0:
        raise ValueError("--train-split and --val-split must be > 0 and sum to less than 1.0.")

    if args.download or args.force_download:
        download_and_extract_dataset(args.dataset_dir, force=args.force_download)

    img_dir, ann_dir, xml_dir, trimap_dir = discover_dataset_paths(args.dataset_dir)
    species_map = load_species_map(ann_dir)

    breed_groups = defaultdict(list)
    for xml_file in sorted(xml_dir.glob("*.xml")):
        stem = xml_file.stem
        img_file = img_dir / f"{stem}.jpg"
        trimap_file = trimap_dir / f"{stem}.png"

        if img_file.exists() and trimap_file.exists():
            breed = "_".join(stem.split("_")[:-1])
            species = get_species(stem, species_map)
            breed_groups[breed].append((species, img_file, xml_file, trimap_file))

    total_imgs = sum(len(v) for v in breed_groups.values())
    if total_imgs == 0:
        raise RuntimeError("No matching (image, xml, trimap) triplets found.")

    rng = random.Random(args.seed)
    counts = defaultdict(int)
    manifest_rows = []
    error_count = 0

    for breed, triplets in sorted(breed_groups.items()):
        rng.shuffle(triplets)
        n = len(triplets)
        train_end = int(n * args.train_split)
        val_end = train_end + int(n * args.val_split)

        splits = {
            "train": triplets[:train_end],
            "val": triplets[train_end:val_end],
            "test": triplets[val_end:],
        }

        for split_name, items in splits.items():
            for species, img_path, xml_path, trimap_path in items:
                out_file = args.output_dir / split_name / species / f"{img_path.stem}_mooney.png"
                ok, msg = process_single_face(
                    img_path=img_path,
                    xml_path=xml_path,
                    trimap_path=trimap_path,
                    out_path=out_file,
                    target_size=args.size,
                    blur_strength=args.blur,
                    white_ratio=args.white_ratio,
                    pad_ratio=args.pad_ratio,
                    bg_mode=args.bg_mode,
                    rng=rng,
                )
                if ok:
                    counts[f"{split_name}/{species}"] += 1
                    manifest_rows.append([split_name, species, breed, img_path.stem, str(out_file)])

                    if args.save_paired_rgb:
                        rgb = cv2.imread(str(img_path))
                        raw_bbox = parse_head_bbox(xml_path)
                        xmin, ymin, xmax, ymax = pad_and_clamp_bbox(
                            raw_bbox, rgb.shape[1], rgb.shape[0], args.pad_ratio
                        )
                        rgb_crop = cv2.resize(rgb[ymin:ymax, xmin:xmax], (args.size, args.size))
                        rgb_out = args.output_dir / f"{split_name}_rgb" / species / f"{img_path.stem}.png"
                        rgb_out.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(rgb_out), rgb_crop)
                else:
                    error_count += 1
                    print(f"[WARN] {msg}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "split_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "species", "breed", "image_id", "mooney_path"])
        writer.writerows(manifest_rows)

    print("\nDataset Generation Summary (70% Train / 15% Val / 15% Test):")
    for split_name in ["train", "val", "test"]:
        c_cnt = counts[f"{split_name}/cat"]
        d_cnt = counts[f"{split_name}/dog"]
        print(f"  {split_name:<6} -> cat: {c_cnt:<4} | dog: {d_cnt:<4} | total: {c_cnt + d_cnt}")
    print(f"  Skipped -> {error_count} images")
    print(f"  Manifest-> {manifest_path.resolve()}")


if __name__ == "__main__":
    main()