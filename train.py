import os
import csv
import time
from pathlib import Path

from ultralytics import YOLO

# ==================== CONFIGURATION ====================
# Dataset (Roboflow)
ROBOFLOW_WORKSPACE = "gilbert-janong-s-workspace"  # from the dataset URL
ROBOFLOW_PROJECT = "penship-final-project-zjfsn"      # from the dataset URL
ROBOFLOW_VERSION = 7                   # dataset version number on Roboflow
PROJECT_DIR = ""
DATASET_DIR = "dataset"                # download destination folder
CSV_PATH = "experiment_results.csv"

IMGSZ = 640
SEED = 0
PATIENCE = 15
WORKERS = 4

# Augmentasi awal: mengikuti konfigurasi yang ingin diuji
AUG_BASE = {
    "hsv_v": 0.4,
    "hsv_s": 0.3,
    "scale": 0.5,
    "mosaic": 1.0,
    "degrees": 15.0,
    "perspective": 0.001,
    "fliplr": 0.5,
}

# Augmentasi lebih ringan sebagai pembanding
AUG_LIGHT = {
    "hsv_v": 0.2,
    "hsv_s": 0.2,
    "scale": 0.2,
    "mosaic": 0.5,
    "degrees": 5.0,
    "perspective": 0.0,
    "fliplr": 0.5,
}

# ==================== EXPERIMENTS ====================
# Satu faktor diubah dari baseline pada setiap eksperimen,
# kecuali eksperimen model n vs s yang memang membandingkan model.

BASE = {
    "model": "yolov8n.pt",
    "epochs": 50,
    "batch": 16,
    "lr0": 0.001,
    "optimizer": "AdamW",
    "augment": AUG_BASE,
}

EXPERIMENTS = [
    {
        "name": "01_baseline",
        **BASE,
    },
    {
        "name": "02_lr_low",
        **{**BASE, "lr0": 0.0005},
    },
    {
        "name": "03_lr_high",
        **{**BASE, "lr0": 0.002},
    },
    {
        "name": "04_optimizer_sgd",
        **{**BASE, "optimizer": "SGD"},
    },
    {
        "name": "05_batch_8",
        **{**BASE, "batch": 8},
    },
    {
        "name": "06_epochs_100",
        **{**BASE, "epochs": 100},
    },
    {
        "name": "07_augment_light",
        **{**BASE, "augment": AUG_LIGHT},
    },
    {
        "name": "08_model_yolov8s",
        **{**BASE, "model": "yolov8s.pt"},
    },
    {
        "name": "09_no_mosaic",
        **{
            **BASE,
            "augment": {**AUG_BASE, "mosaic": 0.0},
        },
    },
]
# ======================================================


def get_dataset() -> Path:
    """Download dataset only if data.yaml is missing."""
    data_yaml = Path(DATASET_DIR) / "data.yaml"

    if data_yaml.exists():
        print(f"Dataset already exists: {data_yaml}")
        return data_yaml

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set.")

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    version = (
        rf.workspace(ROBOFLOW_WORKSPACE)
        .project(ROBOFLOW_PROJECT)
        .version(ROBOFLOW_VERSION)
    )

    dataset = version.download("yolov8", location=DATASET_DIR)
    data_yaml = Path(dataset.location) / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    return data_yaml


def run_experiment(data_path: Path, config: dict) -> dict:
    """Train one configuration and evaluate on validation only."""
    name = config["name"]
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {name}")
    print(f"Model: {config['model']}")
    print(f"Epochs={config['epochs']} | Batch={config['batch']}")
    print(
        f"LR0={config['lr0']} | "
        f"Optimizer={config['optimizer']}"
    )
    print(f"Augmentation: {config['augment']}")

    model = YOLO(config["model"])
    start = time.perf_counter()

    model.train(
        data=str(data_path),
        epochs=config["epochs"],
        imgsz=IMGSZ,
        batch=config["batch"],
        seed=SEED,
        patience=PATIENCE,
        lr0=config["lr0"],
        optimizer=config["optimizer"],
        workers=WORKERS,

        # Konsistensi eksperimen
        project=PROJECT_DIR,
        name=name,
        exist_ok=False,
        plots=True,
        save=True,
        val=True,

        # Pertahankan pengaturan ini sama untuk semua eksperimen
        amp=True,
        cache=False,
        cos_lr=False,

        # Augmentasi yang sedang diuji
        **config["augment"],
    )

    elapsed = time.perf_counter() - start
    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"

    if not best_path.exists():
        raise FileNotFoundError(
            f"best.pt tidak ditemukan: {best_path}"
        )

    # Evaluasi seluruh konfigurasi dengan validation set
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(data_path),
        split="val",
        imgsz=IMGSZ,
        plots=True,
        verbose=False,
    )

    result = {
        "name": name,
        "model": config["model"],
        "epochs_requested": config["epochs"],
        "epochs_completed": model.trainer.epoch + 1
        if hasattr(model.trainer, "epoch") else "",
        "batch": config["batch"],
        "lr0": config["lr0"],
        "optimizer": config["optimizer"],
        "augmentation": str(config["augment"]),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "training_seconds": round(elapsed, 2),
        "best_model": str(best_path),
    }

    print(
        f"\n{name}: "
        f"mAP50={result['mAP50']:.4f} | "
        f"mAP50-95={result['mAP50_95']:.4f} | "
        f"Precision={result['precision']:.4f} | "
        f"Recall={result['recall']:.4f} | "
        f"Time={elapsed:.1f}s"
    )

    return result


def save_results(results: list):
    """Save experiment results into a CSV table."""
    if not results:
        return

    output = Path(CSV_PATH)
    fields = list(results[0].keys())

    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nExperiment table saved: {output.resolve()}")


def evaluate_final_test(data_path: Path, best_result: dict):
    """Evaluate only the configuration selected using validation."""
    yaml_text = data_path.read_text(encoding="utf-8")

    # Do not accidentally evaluate validation data as test data.
    has_test = any(
        line.strip().startswith("test:")
        for line in yaml_text.splitlines()
    )

    if not has_test:
        print("\nTest split tidak ditemukan di data.yaml.")
        print("Evaluasi akhir dilewati; periksa konfigurasi dataset.")
        return

    model = YOLO(best_result["best_model"])
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=IMGSZ,
        plots=True,
    )

    print("\n===== FINAL TEST RESULT =====")
    print(f"Selected experiment: {best_result['name']}")
    print(f"Test precision: {metrics.box.mp:.4f}")
    print(f"Test recall:    {metrics.box.mr:.4f}")
    print(f"Test mAP50:     {metrics.box.map50:.4f}")
    print(f"Test mAP50-95:  {metrics.box.map:.4f}")


def main():
    data_path = get_dataset()
    results = []

    for config in EXPERIMENTS:
        result = run_experiment(data_path, config)
        results.append(result)

        # Save after every experiment so completed results remain available.
        save_results(results)

    # Choose the model using validation mAP50-95 only.
    best_result = max(results, key=lambda row: row["mAP50_95"])

    print("\n===== BEST VALIDATION MODEL =====")
    print(f"Experiment: {best_result['name']}")
    print(f"Weights: {best_result['best_model']}")
    print(f"Validation mAP50-95: {best_result['mAP50_95']:.4f}")

    # Use the test set only after configuration selection.
    evaluate_final_test(data_path, best_result)


if __name__ == "__main__":
    main()