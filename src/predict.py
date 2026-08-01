"""Classify fabric images with the trained ResNet18 model.

Usage:
    python src/predict.py image.jpg
    python src/predict.py folder/ --threshold 0.95
    python src/predict.py image.jpg --all

The model and its settings are read from models/. Nothing is trained here;
see notebooks/04_resnet18_transfer_learning.ipynb for how the weights were made.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

MIN_SIDE = 32
MAX_ASPECT = 6.0
MIN_STD = 4.0  # a flat image has almost no variation in brightness
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class InvalidImage(Exception):
    """The file cannot be classified, and the message explains why."""


def load_model(weights=None):
    config_path = MODELS / "resnet18_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"Settings file not found: {config_path}\n"
            f"Run notebooks/04_resnet18_transfer_learning.ipynb to produce it."
        )
    config = json.load(open(config_path))

    if weights is None:
        for name in config["weight_files"]:
            if (MODELS / name).exists():
                weights = MODELS / name
                break
    if weights is None or not Path(weights).exists():
        raise SystemExit(
            f"No weight file found in {MODELS}.\n"
            f"Expected one of: {', '.join(config['weight_files'])}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(config["class_names"]))
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    prepare = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(config["normalize_mean"], config["normalize_std"]),
    ])
    return model, prepare, config, device, Path(weights)


def load_and_check(path):
    """Open an image, refusing what cannot be read and warning about what looks odd."""
    path = Path(path)

    if not path.exists():
        raise InvalidImage(f"file not found: {path}")
    if path.is_dir():
        raise InvalidImage(f"that is a folder, not an image: {path}")
    if path.stat().st_size == 0:
        raise InvalidImage("file is empty (0 bytes)")

    try:
        with Image.open(path) as probe:
            probe.verify()
    except Exception as exc:
        raise InvalidImage(
            f"not a readable image ({type(exc).__name__}); "
            f"supported formats: JPEG, PNG, BMP, TIFF"
        ) from None

    img = Image.open(path)
    warnings = []
    width, height = img.size

    if min(width, height) < MIN_SIDE:
        warnings.append(f"very small ({width}x{height}); texture will be lost when resized")
    if max(width, height) / max(min(width, height), 1) > MAX_ASPECT:
        warnings.append(f"extreme shape ({width}x{height}); squashing will distort it")
    if img.mode != "RGB":
        warnings.append(f"stored as '{img.mode}', converted to RGB")

    rgb = img.convert("RGB")

    # A nearly flat image carries no fabric texture. The model still answers
    # confidently on these, so the check has to happen before it is asked.
    grey = rgb.convert("L")
    pixels = list(grey.getdata())
    mean = sum(pixels) / len(pixels)
    spread = (sum((p - mean) ** 2 for p in pixels) / len(pixels)) ** 0.5
    if spread < MIN_STD:
        raise InvalidImage(
            f"image is almost featureless (brightness variation {spread:.1f}); "
            f"this does not look like a photograph of fabric"
        )

    return rgb, warnings


def predict(path, model, prepare, config, device, threshold):
    img, warnings = load_and_check(path)

    batch = prepare(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(batch).softmax(dim=1)[0].cpu()

    order = probs.argsort(descending=True)
    best = int(order[0])
    names = config["class_names"]

    return {
        "file": Path(path).name,
        "prediction": names[best],
        "confidence": float(probs[best]),
        "is_defect": names[best] != "defect free",
        "needs_review": float(probs[best]) < threshold or bool(warnings),
        "ranked": [(names[int(i)], float(probs[int(i)])) for i in order],
        "warnings": warnings,
    }


def report(result, show_all=False):
    verdict = "DEFECT" if result["is_defect"] else "no defect"
    print(f"{result['file']}")
    print(f"  {result['prediction']}  ({result['confidence']:.1%})   [{verdict}]")

    shown = result["ranked"] if show_all else result["ranked"][1:3]
    label = "all classes" if show_all else "also considered"
    if show_all:
        print(f"  {label}:")
        for name, p in shown:
            print(f"    {name:16s} {p:6.1%} {'#' * int(p * 40)}")
    else:
        for name, p in shown:
            print(f"     {label}: {name} ({p:.1%})")

    for w in result["warnings"]:
        print(f"  WARNING: {w}")
    if result["needs_review"]:
        print("  -> flagged for human review")


def main():
    parser = argparse.ArgumentParser(
        description="Classify a fabric image, or every image in a folder.")
    parser.add_argument("path", help="image file or folder of images")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="confidence below this is flagged for review (default: 0.95)")
    parser.add_argument("--weights", default=None,
                        help="specific weight file to use (default: first available)")
    parser.add_argument("--all", action="store_true",
                        help="show probabilities for every class")
    args = parser.parse_args()

    model, prepare, config, device, weights = load_model(args.weights)
    print(f"model: {weights.name} on {device}\n")

    target = Path(args.path)
    if target.is_dir():
        files = sorted(p for p in target.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"No image files found in {target}")
        reviewed = defects = refused = 0
        for path in files:
            try:
                result = predict(path, model, prepare, config, device, args.threshold)
                report(result, args.all)
                reviewed += result["needs_review"]
                defects += result["is_defect"]
            except InvalidImage as exc:
                print(f"{path.name}\n  REFUSED: {exc}")
                refused += 1
            print()
        print(f"{len(files)} files | {defects} defects | "
              f"{reviewed} need review | {refused} refused")
    else:
        try:
            report(predict(target, model, prepare, config, device, args.threshold),
                   args.all)
        except InvalidImage as exc:
            print(f"{target.name}\n  REFUSED: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
