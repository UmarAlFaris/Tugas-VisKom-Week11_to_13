"""Training and evaluation helpers for the Viskom Week 11–13 assignment.

The notebooks are the intended entry points. This module keeps smoke tests and
full experiments on exactly the same code paths.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


RESULT_COLUMNS = [
    "run_id",
    "model_name",
    "architecture_family",
    "initialization",
    "weights_source",
    "evaluation_dataset",
    "train_images",
    "val_images",
    "test_images",
    "num_classes",
    "seed",
    "input_size",
    "epochs_configured",
    "best_epoch",
    "batch_size",
    "optimizer",
    "learning_rate",
    "metric_backend",
    "mask_map_50_95",
    "mask_map_50",
    "mask_map_75",
    "mask_mar_100",
    "train_time_seconds",
    "peak_vram_mb",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "checkpoint_path",
    "status",
    "timestamp_utc",
    "notes",
]


def experiment_matrix() -> list[dict[str, Any]]:
    """Return the locked six-run experimental matrix."""

    return [
        {
            "run_id": "yolo26n_seg_pretrained",
            "model_key": "yolo26",
            "model_name": "YOLO26n-Seg",
            "architecture_family": "one-stage",
            "initialization": "pretrained",
            "weights_source": "COCO",
            "batch_size": 4,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
        },
        {
            "run_id": "yolo26n_seg_scratch",
            "model_key": "yolo26",
            "model_name": "YOLO26n-Seg",
            "architecture_family": "one-stage",
            "initialization": "scratch",
            "weights_source": "random",
            "batch_size": 4,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
        },
        {
            "run_id": "maskrcnn_r50_fpn_v2_pretrained",
            "model_key": "maskrcnn",
            "model_name": "Mask R-CNN ResNet-50-FPN v2",
            "architecture_family": "two-stage CNN",
            "initialization": "pretrained",
            "weights_source": "COCO",
            "batch_size": 2,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        },
        {
            "run_id": "maskrcnn_r50_fpn_v2_scratch",
            "model_key": "maskrcnn",
            "model_name": "Mask R-CNN ResNet-50-FPN v2",
            "architecture_family": "two-stage CNN",
            "initialization": "scratch",
            "weights_source": "random",
            "batch_size": 2,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        },
        {
            "run_id": "mask2former_swin_tiny_pretrained",
            "model_key": "mask2former",
            "model_name": "Mask2Former (Swin-T backbone)",
            "architecture_family": "transformer-based",
            "initialization": "pretrained",
            "weights_source": "facebook/mask2former-swin-tiny-coco-instance",
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        },
        {
            "run_id": "mask2former_swin_tiny_scratch",
            "model_key": "mask2former",
            "model_name": "Mask2Former (Swin-T backbone)",
            "architecture_family": "transformer-based",
            "initialization": "scratch",
            "weights_source": "random",
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
        },
    ]


def get_run(run_id: str) -> dict[str, Any]:
    matches = [item for item in experiment_matrix() if item["run_id"] == run_id]
    if not matches:
        valid = ", ".join(item["run_id"] for item in experiment_matrix())
        raise KeyError(f"Unknown run_id {run_id!r}. Valid IDs: {valid}")
    return dict(matches[0])


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this notebook kernel. Select the CUDA-enabled "
            "Python 3.11 environment before training."
        )
    return torch.device("cuda:0")


def environment_summary() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "ultralytics": _package_version("ultralytics"),
        "transformers": _package_version("transformers"),
        "torchmetrics": _package_version("torchmetrics"),
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
            if cuda_available
            else None
        ),
    }


def _package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def read_dataset_yaml(dataset_root: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_class_names(dataset_root: str | Path) -> list[str]:
    raw_names = read_dataset_yaml(dataset_root).get("names")
    if isinstance(raw_names, list):
        return [str(value) for value in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(raw_names, key=lambda x: int(x))]
    raise ValueError("data.yaml must contain names as a list or integer-keyed mapping")


def normalize_split(split: str) -> str:
    split = split.lower()
    return "valid" if split in {"val", "validation", "valid"} else split


def list_images(dataset_root: str | Path, split: str) -> list[Path]:
    split = normalize_split(split)
    image_dir = Path(dataset_root) / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in extensions)


def audit_dataset(dataset_root: str | Path) -> dict[str, Any]:
    """Audit split counts, class counts, and basic label integrity."""

    dataset_root = Path(dataset_root)
    names = read_class_names(dataset_root)
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "num_classes": len(names),
        "class_names": names,
        "splits": {},
    }
    class_counts_by_split: dict[str, Counter[int]] = {}

    for split in ("train", "valid", "test"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue

        images = list_images(dataset_root, split)
        labels = sorted(label_dir.glob("*.txt"))
        class_counts: Counter[int] = Counter()
        malformed: list[str] = []
        empty_labels = 0
        instance_count = 0

        for label_path in labels:
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except (ValueError, IndexError):
                    malformed.append(f"{label_path.name}:{line_number}")
                    continue
                if (
                    class_id < 0
                    or class_id >= len(names)
                    or len(coords) < 6
                    or len(coords) % 2 != 0
                    or any(not math.isfinite(value) for value in coords)
                ):
                    malformed.append(f"{label_path.name}:{line_number}")
                    continue
                class_counts[class_id] += 1
                instance_count += 1

        report["splits"][split] = {
            "images": len(images),
            "label_files": len(labels),
            "instances": instance_count,
            "empty_label_files": empty_labels,
            "malformed_annotations": malformed,
            "class_counts": {
                names[class_id]: class_counts.get(class_id, 0)
                for class_id in range(len(names))
            },
        }
        class_counts_by_split[split] = class_counts

    train_counts = class_counts_by_split.get("train", Counter())
    test_counts = class_counts_by_split.get("test", Counter())
    report["classes_missing_from_train_but_in_test"] = [
        names[class_id]
        for class_id in range(len(names))
        if train_counts.get(class_id, 0) == 0 and test_counts.get(class_id, 0) > 0
    ]
    report["classes_with_fewer_than_10_train_instances"] = [
        names[class_id]
        for class_id in range(len(names))
        if 0 < train_counts.get(class_id, 0) < 10
    ]
    return report


class YoloPolygonDataset(Dataset):
    """Read Ultralytics YOLO polygon labels into tensors and binary masks."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        limit: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = normalize_split(split)
        self.names = read_class_names(self.dataset_root)
        self.image_paths = list_images(self.dataset_root, self.split)
        if limit is not None:
            self.image_paths = self.image_paths[: int(limit)]
        self.label_dir = self.dataset_root / self.split / "labels"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        label_path = self.label_dir / f"{image_path.stem}.txt"

        masks: list[np.ndarray] = []
        labels: list[int] = []
        boxes: list[list[float]] = []
        areas: list[float] = []

        lines = []
        if label_path.exists():
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
            if len(coords) % 2 != 0:
                continue
            points = coords.reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * width
            points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * height
            polygon = [(float(x), float(y)) for x, y in points]

            mask_image = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask_image).polygon(polygon, outline=1, fill=1)
            mask = np.asarray(mask_image, dtype=np.uint8).copy()
            ys, xs = np.nonzero(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min, x_max = float(xs.min()), float(xs.max() + 1)
            y_min, y_max = float(ys.min()), float(ys.max() + 1)
            masks.append(mask)
            labels.append(class_id)
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append(float(mask.sum()))

        if masks:
            masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
        else:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.uint8),
        }
        image_tensor = pil_to_tensor(image).float().div(255.0)
        return image_tensor, target


def detection_collate(batch: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def build_instance_map(
    target: dict[str, torch.Tensor],
) -> tuple[np.ndarray, dict[int, int]]:
    """Create an instance-ID map for Mask2Former.

    Instance ID 0 is background/ignore. Real instances start at 1 and map to
    zero-based semantic class IDs.
    """

    masks = target["masks"].cpu().numpy().astype(bool)
    labels = target["labels"].cpu().tolist()
    if masks.shape[0] == 0:
        height, width = masks.shape[-2:]
    else:
        height, width = masks.shape[1:]
    instance_map = np.zeros((height, width), dtype=np.int32)
    mapping: dict[int, int] = {}
    for index, (mask, label) in enumerate(zip(masks, labels), start=1):
        instance_map[mask] = index
        mapping[index] = int(label)
    return instance_map, mapping


def write_runtime_yolo_yaml(
    dataset_root: str | Path,
    output_dir: str | Path,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> Path:
    """Write a correct absolute-path YAML without altering the downloaded YAML."""

    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = read_class_names(dataset_root)
    limits = {"train": train_limit, "val": val_limit, "test": test_limit}
    normalized = {"val": "valid"}
    split_values: dict[str, str] = {}

    for yaml_key in ("train", "val", "test"):
        folder = normalized.get(yaml_key, yaml_key)
        limit = limits[yaml_key]
        images = list_images(dataset_root, folder)
        if limit is None:
            split_values[yaml_key] = str((dataset_root / folder / "images").resolve())
        else:
            list_path = output_dir / f"{yaml_key}_images.txt"
            selected = images[: int(limit)]
            list_path.write_text(
                "\n".join(str(path.resolve()) for path in selected) + "\n",
                encoding="utf-8",
            )
            split_values[yaml_key] = str(list_path.resolve())

    runtime = {
        **split_values,
        "nc": len(names),
        "names": {index: name for index, name in enumerate(names)},
    }
    output_path = output_dir / "runtime_data.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime, handle, sort_keys=False, allow_unicode=True)
    return output_path


def _new_map_metric():
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    return MeanAveragePrecision(
        iou_type="segm",
        class_metrics=True,
        max_detection_thresholds=[1, 10, 100],
    )


def _empty_prediction(height: int, width: int) -> dict[str, torch.Tensor]:
    return {
        "masks": torch.zeros((0, height, width), dtype=torch.bool),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }


def _target_for_metric(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "masks": target["masks"].bool().cpu(),
        "labels": target["labels"].long().cpu(),
    }


def _scalar_metric(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def simplify_metrics(raw: dict[str, Any]) -> dict[str, float | None]:
    return {
        "mask_map_50_95": _scalar_metric(raw.get("map")),
        "mask_map_50": _scalar_metric(raw.get("map_50")),
        "mask_map_75": _scalar_metric(raw.get("map_75")),
        "mask_mar_100": _scalar_metric(raw.get("mar_100")),
    }


def extract_per_class_metrics(
    raw: dict[str, Any], class_names: list[str]
) -> list[dict[str, Any]]:
    classes = raw.get("classes")
    map_per_class = raw.get("map_per_class")
    mar_per_class = raw.get("mar_100_per_class")
    if not isinstance(classes, torch.Tensor) or not isinstance(map_per_class, torch.Tensor):
        return []
    classes = classes.detach().cpu().tolist()
    maps = map_per_class.detach().cpu().tolist()
    mars = (
        mar_per_class.detach().cpu().tolist()
        if isinstance(mar_per_class, torch.Tensor)
        else [None] * len(classes)
    )
    rows = []
    for class_id, map_value, mar_value in zip(classes, maps, mars):
        class_id = int(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "mask_map_50_95": float(map_value),
                "mask_mar_100": None if mar_value is None else float(mar_value),
            }
        )
    return rows


def evaluate_yolo_model(
    model: Any,
    dataset: YoloPolygonDataset,
    input_size: int,
    device: torch.device,
    score_threshold: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch.nn.functional as functional

    metric = _new_map_metric()
    for index, image_path in enumerate(dataset.image_paths):
        _, target = dataset[index]
        height, width = target["masks"].shape[-2:]
        result = model.predict(
            source=str(image_path),
            imgsz=input_size,
            conf=score_threshold,
            iou=0.7,
            max_det=100,
            device=device.index or 0,
            retina_masks=True,
            verbose=False,
        )[0]

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            prediction = _empty_prediction(height, width)
        else:
            masks = result.masks.data.detach().float()
            if tuple(masks.shape[-2:]) != (height, width):
                masks = functional.interpolate(
                    masks.unsqueeze(1),
                    size=(height, width),
                    mode="nearest",
                ).squeeze(1)
            prediction = {
                "masks": (masks >= 0.5).cpu(),
                "scores": result.boxes.conf.detach().float().cpu(),
                "labels": result.boxes.cls.detach().long().cpu(),
            }
        metric.update([prediction], [_target_for_metric(target)])

    raw = metric.compute()
    return raw, extract_per_class_metrics(raw, dataset.names)


def build_maskrcnn(
    num_classes: int,
    pretrained: bool,
    input_size: int,
) -> torch.nn.Module:
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model_num_classes = num_classes + 1
    if pretrained:
        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=5,
            min_size=input_size,
            max_size=input_size,
        )
        box_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            box_features, model_num_classes
        )
        mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            mask_features, 256, model_num_classes
        )
    else:
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=model_num_classes,
            min_size=input_size,
            max_size=input_size,
        )
    return model


def _maskrcnn_targets_for_model(
    targets: Iterable[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    prepared = []
    allowed = {"boxes", "labels", "masks", "image_id", "area", "iscrowd"}
    for target in targets:
        converted = {
            key: value.to(device)
            for key, value in target.items()
            if key in allowed
        }
        converted["labels"] = converted["labels"] + 1
        prepared.append(converted)
    return prepared


@torch.inference_mode()
def evaluate_maskrcnn_model(
    model: torch.nn.Module,
    dataset: YoloPolygonDataset,
    device: torch.device,
    batch_size: int = 1,
    score_threshold: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric = _new_map_metric()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )
    model.eval()
    for images, targets in loader:
        images_device = [image.to(device) for image in images]
        outputs = model(images_device)
        predictions = []
        for output, target in zip(outputs, targets):
            height, width = target["masks"].shape[-2:]
            keep = output["scores"] >= score_threshold
            if int(keep.sum()) == 0:
                predictions.append(_empty_prediction(height, width))
                continue
            predictions.append(
                {
                    "masks": (output["masks"][keep, 0] >= 0.5).cpu(),
                    "scores": output["scores"][keep].float().cpu(),
                    "labels": (output["labels"][keep].long() - 1).cpu(),
                }
            )
        metric.update(
            predictions,
            [_target_for_metric(target) for target in targets],
        )
    raw = metric.compute()
    return raw, extract_per_class_metrics(raw, dataset.names)


def train_maskrcnn(
    run: dict[str, Any],
    dataset_root: str | Path,
    run_dir: str | Path,
    epochs: int,
    input_size: int,
    seed: int,
    eval_every: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> dict[str, Any]:
    device = get_device()
    seed_everything(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = YoloPolygonDataset(dataset_root, "train", train_limit)
    val_dataset = YoloPolygonDataset(dataset_root, "valid", val_limit)
    names = train_dataset.names
    model = build_maskrcnn(
        num_classes=len(names),
        pretrained=run["initialization"] == "pretrained",
        input_size=input_size,
    ).to(device)

    loader = DataLoader(
        train_dataset,
        batch_size=int(run["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run["learning_rate"]),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    best_map = float("-inf")
    best_epoch = 0
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, targets in loader:
            images_device = [image.to(device) for image in images]
            targets_device = _maskrcnn_targets_for_model(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                losses = model(images_device, targets_device)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite Mask R-CNN loss at epoch {epoch}: {loss.item()}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            batch_count += 1
        scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batch_count),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        should_evaluate = epoch % max(1, eval_every) == 0 or epoch == epochs
        if should_evaluate:
            raw_metrics, _ = evaluate_maskrcnn_model(
                model, val_dataset, device, batch_size=1
            )
            val_map = _scalar_metric(raw_metrics.get("map"))
            row.update(simplify_metrics(raw_metrics))
            if val_map is not None and val_map > best_map:
                best_map = val_map
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "num_classes": len(names),
                        "input_size": input_size,
                        "epoch": epoch,
                        "run": run,
                    },
                    checkpoint_path,
                )
        history.append(row)
        _write_dict_rows(run_dir / "training_log.csv", history)

    elapsed = time.perf_counter() - started
    if not checkpoint_path.exists():
        best_epoch = epochs
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_classes": len(names),
                "input_size": input_size,
                "epoch": epochs,
                "run": run,
            },
            checkpoint_path,
        )
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "train_time_seconds": elapsed,
        "peak_vram_mb": peak_vram,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
    }


MASK2FORMER_CHECKPOINT = "facebook/mask2former-swin-tiny-coco-instance"


def build_mask2former(
    class_names: list[str],
    pretrained: bool,
    input_size: int,
):
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )

    id2label = {index: name for index, name in enumerate(class_names)}
    label2id = {name: index for index, name in enumerate(class_names)}
    processor = Mask2FormerImageProcessor.from_pretrained(
        MASK2FORMER_CHECKPOINT,
        ignore_index=0,
        do_reduce_labels=False,
        # Transformers requires longest_edge to be strictly greater than
        # shortest_edge. A one-pixel ceiling preserves the intended 640-pixel
        # target while allowing aspect-ratio-preserving resizing.
        size={"shortest_edge": input_size, "longest_edge": input_size + 1},
    )
    if pretrained:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MASK2FORMER_CHECKPOINT,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        config = Mask2FormerConfig.from_pretrained(MASK2FORMER_CHECKPOINT)
        config.num_labels = len(class_names)
        config.id2label = id2label
        config.label2id = label2id
        model = Mask2FormerForUniversalSegmentation(config)
    return model, processor


def mask2former_collate(batch, processor):
    images, targets = zip(*batch)
    instance_maps = []
    mappings = []
    for target in targets:
        instance_map, mapping = build_instance_map(target)
        instance_maps.append(instance_map)
        mappings.append(mapping)
    encoded = processor(
        images=list(images),
        segmentation_maps=instance_maps,
        instance_id_to_semantic_id=mappings,
        return_tensors="pt",
        do_rescale=False,
    )
    return encoded


def _move_mask2former_batch(batch: Any, device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            moved[key] = [item.to(device) for item in value]
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.inference_mode()
def evaluate_mask2former_model(
    model: torch.nn.Module,
    processor: Any,
    dataset: YoloPolygonDataset,
    device: torch.device,
    score_threshold: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric = _new_map_metric()
    model.eval()
    for index in range(len(dataset)):
        image, target = dataset[index]
        height, width = target["masks"].shape[-2:]
        encoded = processor(
            images=[image],
            return_tensors="pt",
            do_rescale=False,
        )
        encoded = _move_mask2former_batch(encoded, device)
        outputs = model(**encoded)
        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            target_sizes=[(height, width)],
        )[0]
        segmentation = processed.get("segmentation")
        segments_info = processed.get("segments_info", [])
        masks = []
        scores = []
        labels = []
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu()
            for segment in segments_info:
                score = float(segment.get("score", 0.0))
                if score < score_threshold:
                    continue
                segment_mask = segmentation == int(segment["id"])
                if int(segment_mask.sum()) == 0:
                    continue
                masks.append(segment_mask)
                scores.append(score)
                labels.append(int(segment["label_id"]))
        if masks:
            prediction = {
                "masks": torch.stack(masks).bool(),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        else:
            prediction = _empty_prediction(height, width)
        metric.update([prediction], [_target_for_metric(target)])
    raw = metric.compute()
    return raw, extract_per_class_metrics(raw, dataset.names)


def train_mask2former(
    run: dict[str, Any],
    dataset_root: str | Path,
    run_dir: str | Path,
    epochs: int,
    input_size: int,
    seed: int,
    eval_every: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> dict[str, Any]:
    device = get_device()
    seed_everything(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = YoloPolygonDataset(dataset_root, "train", train_limit)
    val_dataset = YoloPolygonDataset(dataset_root, "valid", val_limit)
    model, processor = build_mask2former(
        train_dataset.names,
        pretrained=run["initialization"] == "pretrained",
        input_size=input_size,
    )
    model.to(device)
    loader = DataLoader(
        train_dataset,
        batch_size=int(run["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=partial(mask2former_collate, processor=processor),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run["learning_rate"]),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    accumulation = int(run.get("gradient_accumulation_steps", 1))
    checkpoint_dir = run_dir / "best_checkpoint"
    best_map = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batch_count = 0
        for step, batch in enumerate(loader, start=1):
            batch_device = _move_mask2former_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch_device)
                loss = outputs.loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite Mask2Former loss at epoch {epoch}: {loss.item()}"
                )
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu()) * accumulation
            batch_count += 1
        scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batch_count),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        should_evaluate = epoch % max(1, eval_every) == 0 or epoch == epochs
        if should_evaluate:
            raw_metrics, _ = evaluate_mask2former_model(
                model, processor, val_dataset, device
            )
            val_map = _scalar_metric(raw_metrics.get("map"))
            row.update(simplify_metrics(raw_metrics))
            if val_map is not None and val_map > best_map:
                best_map = val_map
                best_epoch = epoch
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
        history.append(row)
        _write_dict_rows(run_dir / "training_log.csv", history)

    elapsed = time.perf_counter() - started
    if not checkpoint_dir.exists():
        best_epoch = epochs
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()
    return {
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "best_epoch": best_epoch,
        "train_time_seconds": elapsed,
        "peak_vram_mb": peak_vram,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
    }


def train_yolo(
    run: dict[str, Any],
    dataset_root: str | Path,
    run_dir: str | Path,
    epochs: int,
    input_size: int,
    seed: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    device = get_device()
    seed_everything(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = write_runtime_yolo_yaml(
        dataset_root,
        run_dir,
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=val_limit,
    )
    model_source = (
        "yolo26n-seg.pt"
        if run["initialization"] == "pretrained"
        else "yolo26n-seg.yaml"
    )
    model = YOLO(model_source)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = model.train(
        data=str(runtime_yaml),
        epochs=epochs,
        imgsz=input_size,
        batch=int(run["batch_size"]),
        optimizer=str(run["optimizer"]),
        lr0=float(run["learning_rate"]),
        device=device.index or 0,
        workers=0,
        seed=seed,
        deterministic=True,
        # A YAML model is random initialization; setting this explicitly keeps
        # Ultralytics args.yaml from misleadingly logging its default True.
        pretrained=run["initialization"] == "pretrained",
        amp=True,
        patience=epochs,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    save_dir = Path(result.save_dir)
    checkpoint_path = save_dir / "weights" / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = save_dir / "weights" / "last.pt"
    best_epoch = _best_yolo_epoch(save_dir / "results.csv")
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "train_time_seconds": elapsed,
        "peak_vram_mb": peak_vram,
        "train_images": len(
            YoloPolygonDataset(dataset_root, "train", train_limit)
        ),
        "val_images": len(YoloPolygonDataset(dataset_root, "valid", val_limit)),
    }


def _best_yolo_epoch(results_path: Path) -> int | None:
    if not results_path.exists():
        return None
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    candidate_keys = [
        key
        for key in rows[0]
        if "mAP50-95(M)" in key or ("map50-95" in key.lower() and "(m)" in key.lower())
    ]
    if not candidate_keys:
        return int(float(rows[-1].get("epoch", len(rows) - 1))) + 1
    key = candidate_keys[0]
    best = max(rows, key=lambda row: float(row.get(key, "-inf") or "-inf"))
    return int(float(best.get("epoch", 0))) + 1


def train_experiment(
    run_id: str,
    dataset_root: str | Path,
    artifact_root: str | Path,
    epochs: int = 50,
    input_size: int = 640,
    seed: int = 42,
    eval_every: int = 5,
    train_limit: int | None = None,
    val_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train one locked experiment and save its metadata."""

    run = get_run(run_id)
    run_dir = Path(artifact_root) / run_id
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"{metadata_path} already exists. Set overwrite=True only if you "
            "intentionally want to replace this run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        run=run,
        dataset_root=dataset_root,
        run_dir=run_dir,
        epochs=epochs,
        input_size=input_size,
        seed=seed,
        train_limit=train_limit,
        val_limit=val_limit,
    )
    if run["model_key"] == "yolo26":
        metadata = train_yolo(**common)
    elif run["model_key"] == "maskrcnn":
        metadata = train_maskrcnn(**common, eval_every=eval_every)
    elif run["model_key"] == "mask2former":
        metadata = train_mask2former(**common, eval_every=eval_every)
    else:
        raise ValueError(f"Unsupported model key: {run['model_key']}")

    metadata.update(
        {
            "run": run,
            "epochs_configured": epochs,
            "input_size": input_size,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment_summary(),
        }
    )
    _write_json(metadata_path, metadata)
    return metadata


def load_run_metadata(
    artifact_root: str | Path, run_id: str
) -> dict[str, Any]:
    path = Path(artifact_root) / run_id / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Run metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_saved_experiment(
    run_id: str,
    artifact_root: str | Path,
    evaluation_root: str | Path,
    evaluation_split: str,
    evaluation_name: str,
    results_csv: str | Path,
    score_threshold: float = 0.001,
    limit: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Evaluate one saved checkpoint and upsert its consolidated CSV row."""

    from ultralytics import YOLO

    device = get_device()
    metadata = load_run_metadata(artifact_root, run_id)
    run = metadata["run"]
    checkpoint_path = Path(metadata["checkpoint_path"])
    dataset = YoloPolygonDataset(evaluation_root, evaluation_split, limit)
    input_size = int(metadata["input_size"])

    if run["model_key"] == "yolo26":
        model = YOLO(str(checkpoint_path))
        raw, per_class = evaluate_yolo_model(
            model, dataset, input_size, device, score_threshold
        )
    elif run["model_key"] == "maskrcnn":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_maskrcnn(
            num_classes=len(dataset.names),
            pretrained=False,
            input_size=input_size,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        raw, per_class = evaluate_maskrcnn_model(
            model, dataset, device, batch_size=1, score_threshold=score_threshold
        )
    elif run["model_key"] == "mask2former":
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_path
        ).to(device)
        processor = Mask2FormerImageProcessor.from_pretrained(checkpoint_path)
        raw, per_class = evaluate_mask2former_model(
            model, processor, dataset, device, score_threshold
        )
    else:
        raise ValueError(f"Unsupported model key: {run['model_key']}")

    metrics = simplify_metrics(raw)
    evaluation_dir = (
        Path(artifact_root) / run_id / "evaluation" / evaluation_name
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_dict_rows(evaluation_dir / "per_class_metrics.csv", per_class)
    _write_json(
        evaluation_dir / "summary.json",
        {
            "run_id": run_id,
            "evaluation_dataset": evaluation_name,
            "test_images": len(dataset),
            **metrics,
        },
    )

    env = metadata.get("environment", {})
    row = {
        "run_id": run_id,
        "model_name": run["model_name"],
        "architecture_family": run["architecture_family"],
        "initialization": run["initialization"],
        "weights_source": run["weights_source"],
        "evaluation_dataset": evaluation_name,
        "train_images": metadata.get("train_images"),
        "val_images": metadata.get("val_images"),
        "test_images": len(dataset),
        "num_classes": len(dataset.names),
        "seed": metadata.get("seed"),
        "input_size": input_size,
        "epochs_configured": metadata.get("epochs_configured"),
        "best_epoch": metadata.get("best_epoch"),
        "batch_size": run.get("batch_size"),
        "optimizer": run.get("optimizer"),
        "learning_rate": run.get("learning_rate"),
        "metric_backend": "TorchMetrics COCO-style segmentation mAP",
        **metrics,
        "train_time_seconds": metadata.get("train_time_seconds"),
        "peak_vram_mb": metadata.get("peak_vram_mb"),
        "gpu_name": env.get("gpu"),
        "torch_version": env.get("torch"),
        "cuda_version": env.get("torch_cuda"),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    upsert_result(results_csv, row)
    del model
    torch.cuda.empty_cache()
    return row


def upsert_result(results_csv: str | Path, row: dict[str, Any]) -> None:
    """Insert or replace one run/evaluation row while preserving CSV schema."""

    results_csv = Path(results_csv)
    existing: list[dict[str, Any]] = []
    if results_csv.exists() and results_csv.stat().st_size > 0:
        with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    key = (str(row["run_id"]), str(row["evaluation_dataset"]))
    filtered = [
        item
        for item in existing
        if (item.get("run_id"), item.get("evaluation_dataset")) != key
    ]
    normalized = {column: row.get(column, "") for column in RESULT_COLUMNS}
    filtered.append(normalized)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(filtered)


def _write_dict_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def visualize_ground_truth(
    dataset: YoloPolygonDataset,
    index: int = 0,
    alpha: float = 0.45,
):
    """Display one decoded image and its instance masks."""

    import matplotlib.pyplot as plt

    image, target = dataset[index]
    image_np = image.permute(1, 2, 0).numpy()
    overlay = image_np.copy()
    rng = np.random.default_rng(42)
    colors = rng.random((max(1, len(target["masks"])), 3))
    for mask, color in zip(target["masks"].numpy().astype(bool), colors):
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    plt.figure(figsize=(10, 7))
    plt.imshow(np.clip(overlay, 0, 1))
    plt.title(
        f"{dataset.image_paths[index].name}: {len(target['labels'])} instances"
    )
    plt.axis("off")
    plt.show()


def clear_analysis_model_cache(model_cache: dict[str, Any] | None = None) -> None:
    """Release models retained by an analysis notebook."""

    if model_cache is not None:
        model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_analysis_bundle(
    run_id: str,
    artifact_root: str | Path,
    evaluation_root: str | Path,
    model_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one saved experiment for interactive prediction.

    A caller may supply a cache to avoid reloading the currently inspected
    model. Do not cache all six models on an 8 GB GPU.
    """

    if model_cache is not None and run_id in model_cache:
        return model_cache[run_id]

    device = get_device()
    metadata = load_run_metadata(artifact_root, run_id)
    run = metadata["run"]
    checkpoint_path = Path(metadata["checkpoint_path"])
    class_names = read_class_names(evaluation_root)
    processor = None

    if run["model_key"] == "yolo26":
        from ultralytics import YOLO

        model = YOLO(str(checkpoint_path))
    elif run["model_key"] == "maskrcnn":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_maskrcnn(
            num_classes=len(class_names),
            pretrained=False,
            input_size=int(metadata["input_size"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
    elif run["model_key"] == "mask2former":
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_path
        ).to(device).eval()
        processor = Mask2FormerImageProcessor.from_pretrained(checkpoint_path)
    else:
        raise ValueError(f"Unsupported model key: {run['model_key']}")

    bundle = {
        "run_id": run_id,
        "run": run,
        "metadata": metadata,
        "model": model,
        "processor": processor,
        "device": device,
        "class_names": class_names,
    }
    if model_cache is not None:
        model_cache[run_id] = bundle
    return bundle


@torch.inference_mode()
def predict_dataset_item(
    bundle: dict[str, Any],
    dataset: YoloPolygonDataset,
    index: int,
    score_threshold: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run one saved experiment on one dataset item."""

    import torch.nn.functional as functional

    image, target = dataset[index]
    height, width = target["masks"].shape[-2:]
    run = bundle["run"]
    model = bundle["model"]
    device = bundle["device"]

    if run["model_key"] == "yolo26":
        result = model.predict(
            source=str(dataset.image_paths[index]),
            imgsz=int(bundle["metadata"]["input_size"]),
            conf=score_threshold,
            iou=0.7,
            max_det=100,
            device=device.index or 0,
            retina_masks=True,
            verbose=False,
        )[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            prediction = _empty_prediction(height, width)
        else:
            masks = result.masks.data.detach().float()
            if tuple(masks.shape[-2:]) != (height, width):
                masks = functional.interpolate(
                    masks.unsqueeze(1),
                    size=(height, width),
                    mode="nearest",
                ).squeeze(1)
            prediction = {
                "masks": (masks >= 0.5).cpu(),
                "scores": result.boxes.conf.detach().float().cpu(),
                "labels": result.boxes.cls.detach().long().cpu(),
            }
    elif run["model_key"] == "maskrcnn":
        outputs = model([image.to(device)])[0]
        keep = outputs["scores"] >= score_threshold
        if int(keep.sum()) == 0:
            prediction = _empty_prediction(height, width)
        else:
            prediction = {
                "masks": (outputs["masks"][keep, 0] >= 0.5).cpu(),
                "scores": outputs["scores"][keep].float().cpu(),
                "labels": (outputs["labels"][keep].long() - 1).cpu(),
            }
    elif run["model_key"] == "mask2former":
        processor = bundle["processor"]
        encoded = processor(
            images=[image],
            return_tensors="pt",
            do_rescale=False,
        )
        encoded = _move_mask2former_batch(encoded, device)
        outputs = model(**encoded)
        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            target_sizes=[(height, width)],
        )[0]
        segmentation = processed.get("segmentation")
        segments_info = processed.get("segments_info", [])
        masks = []
        scores = []
        labels = []
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu()
            for segment in segments_info:
                score = float(segment.get("score", 0.0))
                if score < score_threshold:
                    continue
                mask = segmentation == int(segment["id"])
                if int(mask.sum()) == 0:
                    continue
                masks.append(mask)
                scores.append(score)
                labels.append(int(segment["label_id"]))
        if masks:
            prediction = {
                "masks": torch.stack(masks).bool(),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        else:
            prediction = _empty_prediction(height, width)
    else:
        raise ValueError(f"Unsupported model key: {run['model_key']}")
    return image, target, prediction


@torch.inference_mode()
def predict_image_file(
    bundle: dict[str, Any],
    image_path: str | Path,
    score_threshold: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one saved experiment on an unannotated image for the live demo."""

    import torch.nn.functional as functional

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Demo image not found: {image_path}")
    with Image.open(image_path) as opened:
        image = pil_to_tensor(opened.convert("RGB")).float() / 255.0
    height, width = image.shape[-2:]
    run = bundle["run"]
    model = bundle["model"]
    device = bundle["device"]

    if run["model_key"] == "yolo26":
        result = model.predict(
            source=str(image_path),
            imgsz=int(bundle["metadata"]["input_size"]),
            conf=score_threshold,
            iou=0.7,
            max_det=100,
            device=device.index or 0,
            retina_masks=True,
            verbose=False,
        )[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            prediction = _empty_prediction(height, width)
        else:
            masks = result.masks.data.detach().float()
            if tuple(masks.shape[-2:]) != (height, width):
                masks = functional.interpolate(
                    masks.unsqueeze(1),
                    size=(height, width),
                    mode="nearest",
                ).squeeze(1)
            prediction = {
                "masks": (masks >= 0.5).cpu(),
                "scores": result.boxes.conf.detach().float().cpu(),
                "labels": result.boxes.cls.detach().long().cpu(),
            }
    elif run["model_key"] == "maskrcnn":
        output = model([image.to(device)])[0]
        keep = output["scores"] >= score_threshold
        if int(keep.sum()) == 0:
            prediction = _empty_prediction(height, width)
        else:
            prediction = {
                "masks": (output["masks"][keep, 0] >= 0.5).cpu(),
                "scores": output["scores"][keep].float().cpu(),
                "labels": (output["labels"][keep].long() - 1).cpu(),
            }
    elif run["model_key"] == "mask2former":
        processor = bundle["processor"]
        encoded = processor(
            images=[image],
            return_tensors="pt",
            do_rescale=False,
        )
        encoded = _move_mask2former_batch(encoded, device)
        outputs = model(**encoded)
        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            target_sizes=[(height, width)],
        )[0]
        segmentation = processed.get("segmentation")
        masks = []
        scores = []
        labels = []
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu()
            for segment in processed.get("segments_info", []):
                score = float(segment.get("score", 0.0))
                mask = segmentation == int(segment["id"])
                if score < score_threshold or int(mask.sum()) == 0:
                    continue
                masks.append(mask)
                scores.append(score)
                labels.append(int(segment["label_id"]))
        prediction = (
            {
                "masks": torch.stack(masks).bool(),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
            if masks
            else _empty_prediction(height, width)
        )
    else:
        raise ValueError(f"Unsupported model key: {run['model_key']}")
    return image, prediction


def match_prediction_instances(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Greedily match predicted and ground-truth masks.

    True positives require both the correct class and mask IoU at or above the
    threshold. A geometrically matched wrong class is recorded separately and
    counts as one false positive plus one false negative in standard counts.
    """

    pred_masks = prediction["masks"].bool().cpu()
    pred_labels = prediction["labels"].long().cpu()
    gt_masks = target["masks"].bool().cpu()
    gt_labels = target["labels"].long().cpu()
    pred_count = len(pred_masks)
    gt_count = len(gt_masks)
    ious = torch.zeros((pred_count, gt_count), dtype=torch.float32)

    for pred_index in range(pred_count):
        if gt_count == 0:
            break
        intersection = (
            pred_masks[pred_index].unsqueeze(0) & gt_masks
        ).sum(dim=(1, 2)).float()
        union = (
            pred_masks[pred_index].unsqueeze(0) | gt_masks
        ).sum(dim=(1, 2)).float()
        ious[pred_index] = torch.where(
            union > 0, intersection / union, torch.zeros_like(union)
        )

    candidates = []
    for pred_index in range(pred_count):
        for gt_index in range(gt_count):
            if (
                pred_labels[pred_index] == gt_labels[gt_index]
                and float(ious[pred_index, gt_index]) >= iou_threshold
            ):
                candidates.append(
                    (
                        float(ious[pred_index, gt_index]),
                        pred_index,
                        gt_index,
                    )
                )
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    true_matches = []
    for iou, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        true_matches.append(
            {"pred_index": pred_index, "gt_index": gt_index, "iou": iou}
        )

    wrong_candidates = []
    for pred_index in range(pred_count):
        if pred_index in used_pred:
            continue
        for gt_index in range(gt_count):
            if gt_index in used_gt:
                continue
            if (
                pred_labels[pred_index] != gt_labels[gt_index]
                and float(ious[pred_index, gt_index]) >= iou_threshold
            ):
                wrong_candidates.append(
                    (
                        float(ious[pred_index, gt_index]),
                        pred_index,
                        gt_index,
                    )
                )
    wrong_candidates.sort(reverse=True)
    wrong_matches = []
    for iou, pred_index, gt_index in wrong_candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        wrong_matches.append(
            {"pred_index": pred_index, "gt_index": gt_index, "iou": iou}
        )

    unmatched_pred = [
        index for index in range(pred_count) if index not in used_pred
    ]
    unmatched_gt = [index for index in range(gt_count) if index not in used_gt]
    tp = len(true_matches)
    fp = pred_count - tp
    fn = gt_count - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    mean_iou = (
        float(np.mean([item["iou"] for item in true_matches]))
        if true_matches
        else None
    )
    return {
        "iou_threshold": iou_threshold,
        "iou_matrix": ious,
        "true_matches": true_matches,
        "wrong_class_matches": wrong_matches,
        "unmatched_prediction_indices": unmatched_pred,
        "unmatched_ground_truth_indices": unmatched_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "wrong_class": len(wrong_matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
    }


def _dominant_failure(match: dict[str, Any], ground_truth_count: int) -> str:
    if ground_truth_count == 0 and match["fp"] == 0:
        return "empty"
    if match["fp"] == 0 and match["fn"] == 0 and match["wrong_class"] == 0:
        return "successful"
    if match["wrong_class"] > 0:
        return "wrong_class"
    if match["fp"] > 0 and match["fn"] > 0:
        return "mixed"
    if match["fn"] > 0:
        return "false_negative"
    if match["fp"] > 0:
        return "false_positive"
    return "other"


def _load_image_metadata(
    metadata_csv: str | Path | None,
) -> dict[str, dict[str, str]]:
    if metadata_csv is None:
        return {}
    path = Path(metadata_csv)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row.get("filename", ""): row
            for row in csv.DictReader(handle)
            if row.get("filename")
        }


def build_per_image_analysis_cache(
    run_id: str,
    artifact_root: str | Path,
    evaluation_root: str | Path,
    evaluation_split: str,
    evaluation_name: str,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    metadata_csv: str | Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Calculate and save per-image error metrics for interactive analysis."""

    output_dir = Path(artifact_root) / run_id / "evaluation" / evaluation_name
    output_path = output_dir / "per_image_metrics.csv"
    if output_path.exists() and not overwrite:
        return output_path

    dataset = YoloPolygonDataset(evaluation_root, evaluation_split, limit)
    bundle = load_analysis_bundle(run_id, artifact_root, evaluation_root)
    image_metadata = _load_image_metadata(metadata_csv)
    rows = []
    try:
        for index, image_path in enumerate(dataset.image_paths):
            _, target, prediction = predict_dataset_item(
                bundle, dataset, index, score_threshold
            )
            match = match_prediction_instances(
                prediction, target, iou_threshold
            )
            gt_class_ids = sorted(set(target["labels"].tolist()))
            pred_class_ids = sorted(set(prediction["labels"].tolist()))
            metadata = image_metadata.get(image_path.name, {})
            rows.append(
                {
                    "image_index": index,
                    "filename": image_path.name,
                    "run_id": run_id,
                    "evaluation_dataset": evaluation_name,
                    "score_threshold": score_threshold,
                    "iou_threshold": iou_threshold,
                    "ground_truth_instances": len(target["labels"]),
                    "predicted_instances": len(prediction["labels"]),
                    "tp": match["tp"],
                    "fp": match["fp"],
                    "fn": match["fn"],
                    "wrong_class": match["wrong_class"],
                    "precision": match["precision"],
                    "recall": match["recall"],
                    "f1": match["f1"],
                    "mean_matched_iou": match["mean_matched_iou"],
                    "mean_confidence": (
                        float(prediction["scores"].mean())
                        if len(prediction["scores"])
                        else None
                    ),
                    "dominant_failure": _dominant_failure(
                        match, len(target["labels"])
                    ),
                    "ground_truth_classes": "|".join(
                        dataset.names[class_id] for class_id in gt_class_ids
                    ),
                    "predicted_classes": "|".join(
                        dataset.names[class_id] for class_id in pred_class_ids
                    ),
                    "is_difficult": metadata.get("is_difficult", ""),
                    "conditions": metadata.get("conditions", ""),
                    "location": metadata.get("location", ""),
                    "notes": metadata.get("notes", ""),
                }
            )
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(
                    f"{run_id}: analyzed {index + 1}/{len(dataset)} "
                    f"{evaluation_name} images"
                )
    finally:
        del bundle
        torch.cuda.empty_cache()
    _write_dict_rows(output_path, rows)
    return output_path


def select_cached_image(
    per_image_metrics_csv: str | Path,
    selection_mode: str = "random",
    class_filter: str | None = None,
    condition_filter: str | None = None,
    specific_filename: str | None = None,
    random_seed: int | None = None,
) -> dict[str, str] | None:
    """Select an image record for manual inspection, or return None."""

    path = Path(per_image_metrics_csv)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if class_filter:
        rows = [
            row
            for row in rows
            if class_filter
            in {
                item
                for value in (
                    row.get("ground_truth_classes", ""),
                    row.get("predicted_classes", ""),
                )
                for item in value.split("|")
                if item
            }
        ]
    if condition_filter:
        condition_lower = condition_filter.lower()
        rows = [
            row
            for row in rows
            if condition_lower in row.get("conditions", "").lower()
        ]
    if specific_filename:
        rows = [
            row for row in rows if row.get("filename") == specific_filename
        ]
    if not rows:
        return None

    def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
        try:
            value = row.get(key, "")
            return float(value) if value not in {"", None} else default
        except ValueError:
            return default

    rng = random.Random(random_seed)
    if selection_mode == "random":
        return rng.choice(rows)
    if selection_mode == "most_false_positives":
        maximum = max(number(row, "fp") for row in rows)
        candidates = [row for row in rows if number(row, "fp") == maximum]
        return rng.choice(candidates) if maximum > 0 else None
    if selection_mode == "most_false_negatives":
        maximum = max(number(row, "fn") for row in rows)
        candidates = [row for row in rows if number(row, "fn") == maximum]
        return rng.choice(candidates) if maximum > 0 else None
    if selection_mode == "wrong_class":
        candidates = [row for row in rows if number(row, "wrong_class") > 0]
        return rng.choice(candidates) if candidates else None
    if selection_mode == "successful":
        candidates = [
            row
            for row in rows
            if number(row, "fp") == 0
            and number(row, "fn") == 0
            and number(row, "ground_truth_instances") > 0
        ]
        return rng.choice(candidates) if candidates else None
    if selection_mode == "lowest_iou":
        with_matches = [
            row for row in rows if row.get("mean_matched_iou", "") not in {"", None}
        ]
        if not with_matches:
            return None
        minimum = min(number(row, "mean_matched_iou", 1.0) for row in with_matches)
        candidates = [
            row
            for row in with_matches
            if number(row, "mean_matched_iou", 1.0) == minimum
        ]
        return rng.choice(candidates)
    if selection_mode == "specific_image":
        return rows[0] if specific_filename else None
    raise ValueError(
        "selection_mode must be random, most_false_positives, "
        "most_false_negatives, lowest_iou, wrong_class, successful, or "
        "specific_image"
    )


def dataset_index_for_filename(
    dataset: YoloPolygonDataset, filename: str
) -> int | None:
    for index, path in enumerate(dataset.image_paths):
        if path.name == filename:
            return index
    return None


def _class_color(class_id: int) -> np.ndarray:
    import matplotlib.pyplot as plt

    return np.asarray(plt.get_cmap("tab20")(class_id % 20)[:3])


def _mask_overlay(
    image: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image.permute(1, 2, 0).cpu().numpy().copy()
    for mask, label in zip(masks.cpu().numpy().astype(bool), labels.tolist()):
        color = _class_color(int(label))
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0, 1)


def _resize_visualization_inputs(
    image: torch.Tensor,
    target: dict[str, torch.Tensor] | None = None,
    prediction: dict[str, torch.Tensor] | None = None,
    max_edge: int = 1280,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
]:
    """Make memory-safe display copies without changing evaluation resolution."""

    import torch.nn.functional as functional

    height, width = image.shape[-2:]
    longest_edge = max(height, width)
    if longest_edge <= max_edge:
        return image, target, prediction

    scale = max_edge / float(longest_edge)
    output_size = (
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    )
    resized_image = functional.interpolate(
        image.unsqueeze(0),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    def resize_instances(
        instances: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor] | None:
        if instances is None:
            return None
        resized = dict(instances)
        masks = instances["masks"]
        if len(masks) == 0:
            resized["masks"] = torch.zeros(
                (0, *output_size), dtype=torch.bool
            )
        else:
            # Resize one instance at a time. A single 4K float mask is about
            # 36 MiB; converting dozens together can exhaust notebook RAM.
            resized["masks"] = torch.stack(
                [
                    functional.interpolate(
                        mask.float().unsqueeze(0).unsqueeze(0),
                        size=output_size,
                        mode="nearest",
                    ).squeeze(0).squeeze(0).bool()
                    for mask in masks
                ]
            )
        return resized

    return resized_image, resize_instances(target), resize_instances(prediction)


def _class_name(class_names: list[str], class_id: int) -> str:
    return (
        class_names[class_id]
        if 0 <= class_id < len(class_names)
        else f"class_{class_id}"
    )


def _annotation_position(
    mask: np.ndarray,
    occupied: list[tuple[float, float]],
) -> tuple[float, float]:
    """Choose a readable in-mask label position with simple overlap avoidance."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    height, width = mask.shape
    base_x = float(np.median(xs))
    base_y = float(np.median(ys))
    min_dx = max(24.0, width * 0.10)
    min_dy = max(18.0, height * 0.055)
    shifts = [
        (0.0, 0.0),
        (0.0, -min_dy),
        (0.0, min_dy),
        (-min_dx, 0.0),
        (min_dx, 0.0),
        (-min_dx, -min_dy),
        (min_dx, min_dy),
        (-min_dx, min_dy),
        (min_dx, -min_dy),
    ]
    for dx, dy in shifts:
        x = float(np.clip(base_x + dx, 8, max(8, width - 8)))
        y = float(np.clip(base_y + dy, 8, max(8, height - 8)))
        if all(
            abs(x - old_x) >= min_dx or abs(y - old_y) >= min_dy
            for old_x, old_y in occupied
        ):
            occupied.append((x, y))
            return x, y
    occupied.append((base_x, base_y))
    return base_x, base_y


def _draw_mask_annotations(
    axis: Any,
    masks: torch.Tensor,
    texts: list[str],
    outline_colors: list[Any],
    fontsize: int = 8,
    max_labels: int = 20,
) -> None:
    """Draw mask boundaries and collision-reduced labels on an image axis."""

    mask_arrays = masks.detach().cpu().numpy().astype(bool)
    occupied: list[tuple[float, float]] = []
    for index, (mask, text, color) in enumerate(
        zip(mask_arrays, texts, outline_colors)
    ):
        if not mask.any():
            continue
        axis.contour(
            mask.astype(np.uint8),
            levels=[0.5],
            colors=[color],
            linewidths=1.2,
            alpha=0.95,
        )
        if index >= max_labels:
            continue
        x, y = _annotation_position(mask, occupied)
        axis.annotate(
            text,
            xy=(x, y),
            xytext=(0, 0),
            textcoords="offset points",
            color="white",
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "black",
                "edgecolor": color,
                "linewidth": 1.2,
                "alpha": 0.78,
            },
        )


def visualize_prediction_analysis(
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
    run_title: str,
    iou_threshold: float = 0.50,
    save_path: str | Path | None = None,
):
    """Render original, ground truth, prediction, and TP/FP/FN overlays."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    match = match_prediction_instances(prediction, target, iou_threshold)
    image, target, prediction = _resize_visualization_inputs(
        image, target, prediction
    )
    assert target is not None and prediction is not None
    image_np = image.permute(1, 2, 0).cpu().numpy()
    gt_overlay = _mask_overlay(
        image, target["masks"], target["labels"]
    )
    pred_overlay = _mask_overlay(
        image, prediction["masks"], prediction["labels"]
    )
    error_overlay = image_np.copy()

    true_pred = {item["pred_index"] for item in match["true_matches"]}
    wrong_pred = {
        item["pred_index"] for item in match["wrong_class_matches"]
    }
    true_gt = {item["gt_index"] for item in match["true_matches"]}
    wrong_gt = {item["gt_index"] for item in match["wrong_class_matches"]}
    for index, mask in enumerate(prediction["masks"].numpy().astype(bool)):
        if index in true_pred:
            color = np.asarray([0.1, 0.8, 0.2])
        elif index in wrong_pred:
            color = np.asarray([0.7, 0.2, 0.9])
        else:
            color = np.asarray([0.95, 0.1, 0.1])
        error_overlay[mask] = 0.45 * error_overlay[mask] + 0.55 * color
    for index, mask in enumerate(target["masks"].numpy().astype(bool)):
        if index in true_gt or index in wrong_gt:
            continue
        color = np.asarray([1.0, 0.8, 0.0])
        error_overlay[mask] = 0.35 * error_overlay[mask] + 0.65 * color

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    panels = [
        (image_np, "Original"),
        (gt_overlay, f"Ground truth ({len(target['labels'])})"),
        (pred_overlay, f"Predictions ({len(prediction['labels'])})"),
        (
            np.clip(error_overlay, 0, 1),
            (
                f"Errors: TP={match['tp']} FP={match['fp']} "
                f"FN={match['fn']} Wrong={match['wrong_class']}"
            ),
        ),
    ]
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")

    gt_labels = [int(label) for label in target["labels"].tolist()]
    _draw_mask_annotations(
        axes[1],
        target["masks"],
        [f"GT: {_class_name(class_names, label)}" for label in gt_labels],
        [_class_color(label) for label in gt_labels],
        fontsize=8,
    )

    pred_labels = [int(label) for label in prediction["labels"].tolist()]
    pred_scores = [float(score) for score in prediction["scores"].tolist()]
    _draw_mask_annotations(
        axes[2],
        prediction["masks"],
        [
            f"Pred: {_class_name(class_names, label)} {score:.2f}"
            for label, score in zip(pred_labels, pred_scores)
        ],
        [_class_color(label) for label in pred_labels],
        fontsize=8,
    )

    true_by_pred = {
        int(item["pred_index"]): item for item in match["true_matches"]
    }
    wrong_by_pred = {
        int(item["pred_index"]): item
        for item in match["wrong_class_matches"]
    }
    error_texts: list[str] = []
    error_colors: list[np.ndarray] = []
    for pred_index, pred_label in enumerate(pred_labels):
        pred_name = _class_name(class_names, pred_label)
        if pred_index in true_by_pred:
            iou = float(true_by_pred[pred_index]["iou"])
            error_texts.append(f"TP: {pred_name} IoU={iou:.2f}")
            error_colors.append(np.asarray([0.1, 0.8, 0.2]))
        elif pred_index in wrong_by_pred:
            gt_index = int(wrong_by_pred[pred_index]["gt_index"])
            gt_label = int(target["labels"][gt_index])
            gt_name = _class_name(class_names, gt_label)
            error_texts.append(f"Wrong: {pred_name} -> {gt_name}")
            error_colors.append(np.asarray([0.7, 0.2, 0.9]))
        else:
            error_texts.append(f"FP: {pred_name}")
            error_colors.append(np.asarray([0.95, 0.1, 0.1]))
    _draw_mask_annotations(
        axes[3],
        prediction["masks"],
        error_texts,
        error_colors,
        fontsize=7,
        max_labels=15,
    )

    unmatched_gt = [
        int(index) for index in match["unmatched_ground_truth_indices"]
    ]
    if unmatched_gt:
        fn_labels = [int(target["labels"][index]) for index in unmatched_gt]
        _draw_mask_annotations(
            axes[3],
            target["masks"][unmatched_gt],
            [f"FN: {_class_name(class_names, label)}" for label in fn_labels],
            [np.asarray([1.0, 0.8, 0.0]) for _ in fn_labels],
            fontsize=7,
            max_labels=15,
        )
    axes[3].legend(
        handles=[
            Patch(color="#1acc33", label="True positive"),
            Patch(color="#f21a1a", label="False positive"),
            Patch(color="#ffcc00", label="False negative"),
            Patch(color="#b333e6", label="Wrong class"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(run_title, fontsize=12, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, match


def visualize_live_prediction(
    image: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
    run_title: str,
    save_path: str | Path | None = None,
):
    """Render an unannotated image beside its predicted instance masks."""

    import matplotlib.pyplot as plt

    image, _, prediction = _resize_visualization_inputs(
        image, prediction=prediction
    )
    assert prediction is not None
    image_np = image.permute(1, 2, 0).cpu().numpy()
    overlay = _mask_overlay(image, prediction["masks"], prediction["labels"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("New photo")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Predictions ({len(prediction['labels'])})")
    for axis in axes:
        axis.axis("off")
    pred_labels = [int(label) for label in prediction["labels"].tolist()]
    _draw_mask_annotations(
        axes[1],
        prediction["masks"],
        [
            f"Pred: {_class_name(class_names, label)} {float(score):.2f}"
            for label, score in zip(pred_labels, prediction["scores"].tolist())
        ],
        [_class_color(label) for label in pred_labels],
        fontsize=9,
    )
    fig.suptitle(run_title, fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def compare_runs_on_image(
    run_ids: list[str],
    artifact_root: str | Path,
    evaluation_root: str | Path,
    evaluation_split: str,
    filename: str,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    save_path: str | Path | None = None,
):
    """Compare multiple saved runs on the exact same image."""

    import matplotlib.pyplot as plt

    dataset = YoloPolygonDataset(evaluation_root, evaluation_split)
    index = dataset_index_for_filename(dataset, filename)
    if index is None:
        print(f"No image named {filename!r} exists in this dataset.")
        return None
    image, target = dataset[index]
    display_image, display_target, _ = _resize_visualization_inputs(
        image, target=target
    )
    assert display_target is not None
    panels = 2 + len(run_ids)
    columns = 3
    rows = math.ceil(panels / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(20, 6.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(display_image.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Original: {filename}")
    axes[1].imshow(
        _mask_overlay(
            display_image,
            display_target["masks"],
            display_target["labels"],
        )
    )
    axes[1].set_title(f"Ground truth ({len(target['labels'])})")
    gt_labels = [int(label) for label in display_target["labels"].tolist()]
    _draw_mask_annotations(
        axes[1],
        display_target["masks"],
        [f"GT: {_class_name(dataset.names, label)}" for label in gt_labels],
        [_class_color(label) for label in gt_labels],
        fontsize=7,
        max_labels=15,
    )
    for axis in axes[:2]:
        axis.axis("off")

    for panel_index, run_id in enumerate(run_ids, start=2):
        bundle = load_analysis_bundle(run_id, artifact_root, evaluation_root)
        try:
            _, _, prediction = predict_dataset_item(
                bundle, dataset, index, score_threshold
            )
            match = match_prediction_instances(
                prediction, target, iou_threshold
            )
            _, _, display_prediction = _resize_visualization_inputs(
                image, prediction=prediction
            )
            assert display_prediction is not None
            axes[panel_index].imshow(
                _mask_overlay(
                    display_image,
                    display_prediction["masks"],
                    display_prediction["labels"],
                )
            )
            axes[panel_index].set_title(
                f"{run_id}\nTP={match['tp']} FP={match['fp']} "
                f"FN={match['fn']} Wrong={match['wrong_class']}",
                fontsize=9,
            )
            pred_labels = [
                int(label) for label in display_prediction["labels"].tolist()
            ]
            _draw_mask_annotations(
                axes[panel_index],
                display_prediction["masks"],
                [
                    (
                        f"Pred: {_class_name(dataset.names, label)} "
                        f"{float(score):.2f}"
                    )
                    for label, score in zip(
                        pred_labels, display_prediction["scores"].tolist()
                    )
                ],
                [_class_color(label) for label in pred_labels],
                fontsize=7,
                max_labels=15,
            )
            axes[panel_index].axis("off")
        finally:
            del bundle
            prediction = None
            display_prediction = None
            gc.collect()
            torch.cuda.empty_cache()
    for axis in axes[panels:]:
        axis.axis("off")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


AUTO_ANALYSIS_MODES = [
    "most_false_positives",
    "most_false_negatives",
    "wrong_class",
    "lowest_iou",
    "highest_iou",
    "highest_f1",
    "successful",
    "random",
]


def _analysis_number(
    row: dict[str, Any], key: str, default: float = 0.0
) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else default
    except (TypeError, ValueError):
        return default


def _select_analysis_rows(
    rows: list[dict[str, Any]],
    selection_mode: str,
    count: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic ranked sample for one analysis category."""

    if count <= 0 or not rows:
        return []
    rng = random.Random(random_seed)
    candidates = list(rows)
    rng.shuffle(candidates)

    if selection_mode == "random":
        return candidates[:count]
    if selection_mode == "most_false_positives":
        candidates = [row for row in candidates if _analysis_number(row, "fp") > 0]
        candidates.sort(
            key=lambda row: (
                -_analysis_number(row, "fp"),
                -_analysis_number(row, "wrong_class"),
                -_analysis_number(row, "fn"),
            )
        )
    elif selection_mode == "most_false_negatives":
        candidates = [row for row in candidates if _analysis_number(row, "fn") > 0]
        candidates.sort(
            key=lambda row: (
                -_analysis_number(row, "fn"),
                -_analysis_number(row, "wrong_class"),
                -_analysis_number(row, "fp"),
            )
        )
    elif selection_mode == "wrong_class":
        candidates = [
            row for row in candidates if _analysis_number(row, "wrong_class") > 0
        ]
        candidates.sort(
            key=lambda row: (
                -_analysis_number(row, "wrong_class"),
                -_analysis_number(row, "fp"),
                -_analysis_number(row, "fn"),
            )
        )
    elif selection_mode == "lowest_iou":
        candidates = [
            row
            for row in candidates
            if row.get("mean_matched_iou", "") not in {"", None}
        ]
        candidates.sort(
            key=lambda row: _analysis_number(row, "mean_matched_iou", 1.0)
        )
    elif selection_mode == "highest_iou":
        candidates = [
            row
            for row in candidates
            if row.get("mean_matched_iou", "") not in {"", None}
            and _analysis_number(row, "tp") > 0
        ]
        candidates.sort(
            key=lambda row: (
                -_analysis_number(row, "mean_matched_iou"),
                -_analysis_number(row, "f1"),
                -_analysis_number(row, "tp"),
            )
        )
    elif selection_mode == "highest_f1":
        candidates = [
            row
            for row in candidates
            if row.get("f1", "") not in {"", None}
            and _analysis_number(row, "tp") > 0
        ]
        candidates.sort(
            key=lambda row: (
                -_analysis_number(row, "f1"),
                -_analysis_number(row, "mean_matched_iou"),
                -_analysis_number(row, "tp"),
            )
        )
    elif selection_mode == "successful":
        candidates = [
            row
            for row in candidates
            if _analysis_number(row, "fp") == 0
            and _analysis_number(row, "fn") == 0
            and _analysis_number(row, "ground_truth_instances") > 0
        ]
        candidates.sort(
            key=lambda row: (
                -_analysis_number(row, "mean_matched_iou"),
                -_analysis_number(row, "ground_truth_instances"),
            )
        )
    else:
        raise ValueError(
            f"Unsupported automatic analysis mode: {selection_mode!r}. "
            f"Choose from {AUTO_ANALYSIS_MODES}."
        )
    return candidates[:count]


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_figure_stem(filename: str) -> str:
    stem = Path(filename).stem
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)


def _aggregate_analysis_rows(
    rows_by_run: dict[str, list[dict[str, str]]]
) -> list[dict[str, Any]]:
    """Aggregate cached metrics by filename for unbiased all-run selection."""

    grouped: dict[str, list[dict[str, str]]] = {}
    for rows in rows_by_run.values():
        for row in rows:
            grouped.setdefault(row.get("filename", ""), []).append(row)
    aggregated = []
    for filename, rows in grouped.items():
        if not filename:
            continue
        matched_ious = [
            _analysis_number(row, "mean_matched_iou")
            for row in rows
            if row.get("mean_matched_iou", "") not in {"", None}
        ]
        f1_values = [
            _analysis_number(row, "f1")
            for row in rows
            if row.get("f1", "") not in {"", None}
        ]
        aggregated.append(
            {
                "filename": filename,
                "tp": sum(_analysis_number(row, "tp") for row in rows),
                "fp": sum(_analysis_number(row, "fp") for row in rows),
                "fn": sum(_analysis_number(row, "fn") for row in rows),
                "wrong_class": sum(
                    _analysis_number(row, "wrong_class") for row in rows
                ),
                "mean_matched_iou": (
                    float(np.mean(matched_ious)) if matched_ious else ""
                ),
                "f1": float(np.mean(f1_values)) if f1_values else "",
                "ground_truth_instances": _analysis_number(
                    rows[0], "ground_truth_instances"
                ),
                "successful_runs": sum(
                    _analysis_number(row, "fp") == 0
                    and _analysis_number(row, "fn") == 0
                    and _analysis_number(row, "ground_truth_instances") > 0
                    for row in rows
                ),
                "conditions": rows[0].get("conditions", ""),
            }
        )
    return aggregated


def export_automatic_analysis_library(
    run_ids: list[str],
    artifact_root: str | Path,
    evaluation_root: str | Path,
    evaluation_split: str,
    evaluation_name: str,
    output_root: str | Path,
    modes: list[str] | None = None,
    samples_per_mode: int = 3,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    random_seed: int = 42,
    overwrite: bool = False,
    include_all_run_comparisons: bool = True,
) -> dict[str, Path]:
    """Export a resumable presentation-ready qualitative analysis library.

    Per-image metric caches must already exist for every requested run. Missing
    modes or caches are recorded instead of stopping the remaining export.
    """

    import matplotlib.pyplot as plt

    modes = list(modes or AUTO_ANALYSIS_MODES)
    invalid_modes = [mode for mode in modes if mode not in AUTO_ANALYSIS_MODES]
    if invalid_modes:
        raise ValueError(f"Unsupported automatic analysis modes: {invalid_modes}")

    artifact_root = Path(artifact_root)
    evaluation_root = Path(evaluation_root)
    output_root = Path(output_root)
    dataset_output_root = output_root / evaluation_name
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    dataset = YoloPolygonDataset(evaluation_root, evaluation_split)
    rows_by_run: dict[str, list[dict[str, str]]] = {}
    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for run_number, run_id in enumerate(run_ids, start=1):
        cache_path = (
            artifact_root
            / run_id
            / "evaluation"
            / evaluation_name
            / "per_image_metrics.csv"
        )
        cached_rows = _read_csv_rows(cache_path)
        rows_by_run[run_id] = cached_rows
        print(f"[{run_number}/{len(run_ids)}] Exporting {run_id}")
        if not cached_rows:
            for mode in modes:
                manifest_rows.append(
                    {
                        "figure_type": "single_model",
                        "run_id": run_id,
                        "selection_mode": mode,
                        "rank": "",
                        "filename": "",
                        "figure_path": "",
                        "status": "missing_cache",
                        "error": str(cache_path),
                    }
                )
            summary_rows.append(
                {"run_id": run_id, "generated": 0, "skipped": 0, "errors": len(modes)}
            )
            continue

        selections: list[tuple[str, int, dict[str, Any]]] = []
        for mode_number, mode in enumerate(modes):
            selected = _select_analysis_rows(
                cached_rows,
                mode,
                samples_per_mode,
                random_seed + run_number * 100 + mode_number,
            )
            if not selected:
                manifest_rows.append(
                    {
                        "figure_type": "single_model",
                        "run_id": run_id,
                        "selection_mode": mode,
                        "rank": "",
                        "filename": "",
                        "figure_path": "",
                        "status": "no_candidate",
                        "error": "",
                    }
                )
            selections.extend((mode, rank, row) for rank, row in enumerate(selected, 1))

        pending = []
        for mode, rank, row in selections:
            figure_path = (
                dataset_output_root
                / run_id
                / mode
                / f"{rank:02d}_{_safe_figure_stem(row['filename'])}.png"
            )
            pending.append((mode, rank, row, figure_path))

        generated = skipped = errors = 0
        bundle = None
        try:
            if any(overwrite or not item[3].exists() for item in pending):
                bundle = load_analysis_bundle(run_id, artifact_root, evaluation_root)
            for mode, rank, row, figure_path in pending:
                status = "skipped_existing"
                error = ""
                if figure_path.exists() and not overwrite:
                    skipped += 1
                else:
                    try:
                        image_index = dataset_index_for_filename(dataset, row["filename"])
                        if image_index is None:
                            raise ValueError("Filename is not present in the evaluation dataset")
                        image, target, prediction = predict_dataset_item(
                            bundle,
                            dataset,
                            image_index,
                            score_threshold=score_threshold,
                        )
                        title = (
                            f"{run_id} | {mode} #{rank} | {row['filename']}\n"
                            f"cached TP={row.get('tp', '')} FP={row.get('fp', '')} "
                            f"FN={row.get('fn', '')} Wrong={row.get('wrong_class', '')} "
                            f"F1={row.get('f1', '')} "
                            f"IoU={row.get('mean_matched_iou', '')} "
                            f"Confidence={row.get('mean_confidence', '')} | "
                            f"{row.get('conditions', '') or 'conditions not tagged'}"
                        )
                        figure, _ = visualize_prediction_analysis(
                            image,
                            target,
                            prediction,
                            dataset.names,
                            run_title=title,
                            iou_threshold=iou_threshold,
                            save_path=figure_path,
                        )
                        plt.close(figure)
                        del figure, image, target, prediction
                        gc.collect()
                        status = "generated"
                        generated += 1
                    except Exception as exc:  # continue the unattended batch
                        plt.close("all")
                        gc.collect()
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                        errors += 1
                        print(f"  ERROR {mode} #{rank} {row['filename']}: {error}")
                manifest_rows.append(
                    {
                        "figure_type": "single_model",
                        "run_id": run_id,
                        "selection_mode": mode,
                        "rank": rank,
                        "filename": row.get("filename", ""),
                        "figure_path": str(figure_path.relative_to(dataset_output_root)),
                        "status": status,
                        "tp": row.get("tp", ""),
                        "fp": row.get("fp", ""),
                        "fn": row.get("fn", ""),
                        "wrong_class": row.get("wrong_class", ""),
                        "f1": row.get("f1", ""),
                        "mean_matched_iou": row.get("mean_matched_iou", ""),
                        "mean_confidence": row.get("mean_confidence", ""),
                        "conditions": row.get("conditions", ""),
                        "error": error,
                    }
                )
        finally:
            if bundle is not None:
                del bundle
            clear_analysis_model_cache()
        summary_rows.append(
            {"run_id": run_id, "generated": generated, "skipped": skipped, "errors": errors}
        )

    if include_all_run_comparisons and rows_by_run:
        aggregate_rows = _aggregate_analysis_rows(rows_by_run)
        comparison_root = dataset_output_root / "comparisons" / "all_six"
        print("Exporting all-run comparison figures")
        for mode_number, mode in enumerate(modes):
            comparison_rows = list(aggregate_rows)
            if mode == "successful":
                comparison_rows = [
                    row for row in comparison_rows if row.get("successful_runs", 0) > 0
                ]
                comparison_rows.sort(key=lambda row: -row.get("successful_runs", 0))
                selected = comparison_rows[:samples_per_mode]
            else:
                selected = _select_analysis_rows(
                    comparison_rows,
                    mode,
                    samples_per_mode,
                    random_seed + 9000 + mode_number,
                )
            if not selected:
                manifest_rows.append(
                    {
                        "figure_type": "all_runs_comparison",
                        "run_id": "ALL",
                        "selection_mode": mode,
                        "rank": "",
                        "filename": "",
                        "figure_path": "",
                        "status": "no_candidate",
                        "error": "",
                    }
                )
                continue
            for rank, row in enumerate(selected, 1):
                figure_path = (
                    comparison_root
                    / mode
                    / f"{rank:02d}_{_safe_figure_stem(row['filename'])}.png"
                )
                status = "skipped_existing"
                error = ""
                if not figure_path.exists() or overwrite:
                    try:
                        figure = compare_runs_on_image(
                            run_ids,
                            artifact_root,
                            evaluation_root,
                            evaluation_split,
                            row["filename"],
                            score_threshold=score_threshold,
                            iou_threshold=iou_threshold,
                            save_path=figure_path,
                        )
                        if figure is None:
                            raise ValueError("Comparison figure was not created")
                        plt.close(figure)
                        status = "generated"
                    except Exception as exc:  # continue the unattended batch
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                        print(f"  ERROR comparison {mode} #{rank}: {error}")
                manifest_rows.append(
                    {
                        "figure_type": "all_runs_comparison",
                        "run_id": "ALL",
                        "selection_mode": mode,
                        "rank": rank,
                        "filename": row.get("filename", ""),
                        "figure_path": str(figure_path.relative_to(dataset_output_root)),
                        "status": status,
                        "fp": row.get("fp", ""),
                        "fn": row.get("fn", ""),
                        "wrong_class": row.get("wrong_class", ""),
                        "f1": row.get("f1", ""),
                        "mean_matched_iou": row.get("mean_matched_iou", ""),
                        "conditions": row.get("conditions", ""),
                        "error": error,
                    }
                )
                clear_analysis_model_cache()

    manifest_path = dataset_output_root / "index.csv"
    summary_path = dataset_output_root / "generation_summary.csv"
    _write_dict_rows(manifest_path, manifest_rows)
    _write_dict_rows(summary_path, summary_rows)
    readme_path = dataset_output_root / "README.md"
    readme_path.write_text(
        "# Automatic qualitative analysis library\n\n"
        f"Dataset: `{evaluation_name}`  \n"
        f"Score threshold: `{score_threshold}`  \n"
        f"Mask IoU threshold: `{iou_threshold}`  \n"
        f"Samples requested per model/mode: `{samples_per_mode}`\n\n"
        "Each single-model PNG contains the original image, ground-truth masks, "
        "predicted masks, and an error overlay. Error colors are **green = true "
        "positive**, **red = false positive**, **yellow = false negative**, and "
        "**purple = wrong class**.\n\n"
        "Use `index.csv` to filter samples and copy representative figures into "
        "the report or presentation. `no_candidate` is expected when a model has "
        "no example matching a category. Automatic categories help discovery, "
        "but the final five failure cases and mechanism explanations still require "
        "manual inspection.\n",
        encoding="utf-8",
    )
    print(f"Automatic analysis library ready: {dataset_output_root}")
    return {
        "root": dataset_output_root,
        "index": manifest_path,
        "summary": summary_path,
        "readme": readme_path,
    }


def plot_training_curves(
    artifact_root: str | Path,
    run_ids: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot available loss and validation mAP histories."""

    import matplotlib.pyplot as plt
    import pandas as pd

    if run_ids is None:
        run_ids = available_completed_runs(artifact_root)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    plotted = 0
    for run_id in run_ids:
        run_dir = Path(artifact_root) / run_id
        log_path = run_dir / "training_log.csv"
        if not log_path.exists():
            log_path = run_dir / "results.csv"
        if not log_path.exists():
            continue
        frame = pd.read_csv(log_path)
        frame.columns = [str(column).strip() for column in frame.columns]
        epoch = (
            frame["epoch"] + 1
            if "epoch" in frame and frame["epoch"].min() == 0
            else frame.get("epoch", pd.Series(range(1, len(frame) + 1)))
        )
        loss_column = next(
            (
                column
                for column in ("train_loss", "train/seg_loss", "train/box_loss")
                if column in frame
            ),
            None,
        )
        map_column = next(
            (
                column
                for column in ("mask_map_50_95", "metrics/mAP50-95(M)")
                if column in frame
            ),
            None,
        )
        if loss_column:
            axes[0].plot(epoch, frame[loss_column], label=run_id)
        if map_column:
            axes[1].plot(epoch, frame[map_column], marker="o", label=run_id)
        plotted += 1
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation mask mAP50-95")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=7)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_results_comparison(
    results_csv: str | Path,
    evaluation_dataset: str = "original_test",
    save_path: str | Path | None = None,
):
    """Plot overall mask mAP for pretrained/scratch runs."""

    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(results_csv)
    frame = frame[frame["evaluation_dataset"] == evaluation_dataset].copy()
    if frame.empty:
        print(f"No results found for {evaluation_dataset!r}.")
        return None
    frame["label"] = (
        frame["model_name"].str.replace(" (Swin-T backbone)", "", regex=False)
        + "\n"
        + frame["initialization"]
    )
    colors = [
        "#2563EB" if value == "pretrained" else "#F59E0B"
        for value in frame["initialization"]
    ]
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(frame["label"], frame["mask_map_50_95"], color=colors)
    axis.set_ylabel("Mask mAP50-95")
    axis.set_title(f"Model comparison on {evaluation_dataset}")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_domain_shift(
    results_csv: str | Path,
    save_path: str | Path | None = None,
):
    """Plot original-to-personal test mAP change for every run."""

    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(results_csv)
    pivot = frame.pivot_table(
        index="run_id",
        columns="evaluation_dataset",
        values="mask_map_50_95",
        aggfunc="first",
    )
    if not {"original_test", "personal_test"}.issubset(pivot.columns):
        print("Domain-shift plot requires both original_test and personal_test.")
        return None
    pivot["delta"] = pivot["personal_test"] - pivot["original_test"]
    pivot = pivot.sort_values("delta")
    colors = ["#DC2626" if value < 0 else "#16A34A" for value in pivot["delta"]]
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.barh(pivot.index, pivot["delta"], color=colors)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Personal test minus original test mask mAP50-95")
    axis.set_title("Domain-shift performance change")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def create_domain_shift_table(
    results_csv: str | Path,
    save_csv: str | Path | None = None,
    save_path: str | Path | None = None,
):
    """Create a presentation-ready original-versus-personal metric table."""

    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(results_csv)
    metric_columns = ["mask_map_50", "mask_map_50_95"]
    required = {"original_test", "personal_test"}
    if not required.issubset(set(frame["evaluation_dataset"])):
        print("The comparison table requires original_test and personal_test rows.")
        return None

    original = frame[frame["evaluation_dataset"] == "original_test"].copy()
    personal = frame[frame["evaluation_dataset"] == "personal_test"].copy()
    original = original[
        ["run_id", "model_name", "initialization", *metric_columns]
    ].rename(
        columns={
            "mask_map_50": "map50_original",
            "mask_map_50_95": "map50_95_original",
        }
    )
    personal = personal[["run_id", *metric_columns]].rename(
        columns={
            "mask_map_50": "map50_personal",
            "mask_map_50_95": "map50_95_personal",
        }
    )
    summary = original.merge(personal, on="run_id", how="inner", validate="one_to_one")
    if summary.empty:
        print("No matching completed runs exist in both evaluation datasets.")
        return None

    summary["map50_delta"] = summary["map50_personal"] - summary["map50_original"]
    summary["map50_95_delta"] = (
        summary["map50_95_personal"] - summary["map50_95_original"]
    )
    summary = summary[
        [
            "run_id",
            "model_name",
            "initialization",
            "map50_original",
            "map50_personal",
            "map50_delta",
            "map50_95_original",
            "map50_95_personal",
            "map50_95_delta",
        ]
    ]

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_csv, index=False)

    if save_path is not None:
        friendly_names = {
            "YOLO26n-Seg": "YOLO26n-Seg",
            "Mask R-CNN ResNet-50-FPN v2": "Mask R-CNN R50-FPN v2",
            "Mask2Former (Swin-T backbone)": "Mask2Former Swin-T",
        }
        row_labels = [
            f"{friendly_names.get(model_name, model_name)} ({initialization})"
            for model_name, initialization in zip(
                summary["model_name"], summary["initialization"]
            )
        ]
        values = []
        for row_label, (_, row) in zip(row_labels, summary.iterrows()):
            values.append(
                [
                    row_label,
                    f"{row['map50_original']:.3f}",
                    f"{row['map50_personal']:.3f}",
                    f"{row['map50_delta']:+.3f}",
                    f"{row['map50_95_original']:.3f}",
                    f"{row['map50_95_personal']:.3f}",
                    f"{row['map50_95_delta']:+.3f}",
                ]
            )
        headers = [
            "Run",
            "mAP50\noriginal",
            "mAP50\npersonal",
            "Delta",
            "mAP50-95\noriginal",
            "mAP50-95\npersonal",
            "Delta",
        ]
        fig, axis = plt.subplots(figsize=(15, 4.9))
        axis.axis("off")
        axis.set_title(
            "All 6 Runs: Original Test vs Personal Test",
            fontsize=16,
            fontweight="bold",
            pad=18,
        )
        artist = axis.table(
            cellText=values,
            colLabels=headers,
            cellLoc="center",
            colLoc="center",
            colWidths=[0.30, 0.115, 0.115, 0.09, 0.13, 0.13, 0.09],
            bbox=[0.01, 0.12, 0.98, 0.76],
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(10)
        for (row_index, column_index), cell in artist.get_celld().items():
            cell.set_edgecolor("#374151")
            cell.set_linewidth(0.8)
            if row_index == 0:
                cell.set_facecolor("#1F4E78")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("#F8FAFC" if row_index % 2 else "#EAF0F6")
                if column_index == 0:
                    cell.get_text().set_ha("left")
                    cell.get_text().set_fontweight("bold")
                if column_index in {3, 6}:
                    metric = (
                        summary.iloc[row_index - 1]["map50_delta"]
                        if column_index == 3
                        else summary.iloc[row_index - 1]["map50_95_delta"]
                    )
                    cell.set_facecolor("#DCFCE7" if metric >= 0 else "#FEE2E2")
                    cell.get_text().set_color("#166534" if metric >= 0 else "#991B1B")
                    cell.get_text().set_fontweight("bold")

        best_map50 = int(summary["map50_personal"].to_numpy().argmax()) + 1
        best_map50_95 = int(summary["map50_95_personal"].to_numpy().argmax()) + 1
        for row_index, column_index in ((best_map50, 2), (best_map50_95, 5)):
            artist[(row_index, column_index)].get_text().set_fontweight("bold")
            artist[(row_index, column_index)].set_facecolor("#BBF7D0")

        fig.text(
            0.5,
            0.045,
            "Delta = personal test - original test. Green indicates a higher personal-test score; red indicates a decrease.",
            ha="center",
            fontsize=10,
            color="#374151",
        )
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return summary


def plot_per_class_metrics(
    artifact_root: str | Path,
    run_id: str,
    evaluation_name: str,
    top_n: int = 20,
    save_path: str | Path | None = None,
):
    """Plot the highest-AP classes available for one saved evaluation."""

    import matplotlib.pyplot as plt
    import pandas as pd

    path = (
        Path(artifact_root)
        / run_id
        / "evaluation"
        / evaluation_name
        / "per_class_metrics.csv"
    )
    if not path.exists() or path.stat().st_size == 0:
        print(f"Per-class metrics do not exist yet: {path}")
        return None
    frame = pd.read_csv(path).sort_values(
        "mask_map_50_95", ascending=False
    ).head(top_n)
    fig, axis = plt.subplots(figsize=(11, max(5, 0.35 * len(frame))))
    axis.barh(
        frame["class_name"][::-1],
        frame["mask_map_50_95"][::-1],
        color="#2563EB",
    )
    axis.set_xlabel("Mask mAP50-95")
    axis.set_title(f"Top {len(frame)} classes: {run_id} on {evaluation_name}")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def available_completed_runs(artifact_root: str | Path) -> list[str]:
    artifact_root = Path(artifact_root)
    return [
        run["run_id"]
        for run in experiment_matrix()
        if (artifact_root / run["run_id"] / "run_metadata.json").exists()
    ]
