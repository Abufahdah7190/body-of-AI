"""Fine-tune bundled YOLOv8n on charger positives and hard negatives."""
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "charger_dataset"
CONFIG = ROOT / ".charger_yolo_config"
CONFIG.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(CONFIG)


def main():
    (DATA / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: charger\n",
        encoding="utf8",
    )
    from ultralytics import YOLO, settings
    import torch

    torch.set_num_threads(2)
    settings.update({"sync": False})
    model = YOLO(str(ROOT / "yolov8n.pt"))
    model.train(
        data=str(DATA / "data.yaml"), epochs=35, imgsz=416, batch=2,
        device="cpu", workers=0, cache=False, amp=False,
        project=str(ROOT / "object_training_runs"), name="charger-hard-negative",
        exist_ok=False, seed=421, deterministic=True, plots=False,
    )
    run = Path(model.trainer.save_dir)
    best = run / "weights" / "best.pt"
    result = YOLO(str(best)).val(
        data=str(DATA / "data.yaml"), split="test", imgsz=416,
        batch=1, device="cpu", workers=0, plots=False,
        project=str(run), name="held_out_test",
    )
    metrics = {key: float(value) for key, value in result.results_dict.items()}
    image_counts = {
        split: sum(1 for p in (DATA / "images" / split).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for split in ("train", "val", "test")
    }
    metrics.update({
        "class": "charger",
        "train_images": image_counts["train"],
        "train_negative_images": sum(1 for p in (DATA / "images" / "train").glob("negative_*.jpg")),
        "validation_images": image_counts["val"],
        "test_images": image_counts["test"],
        "note": "Experimental fine-tuning with hard-negative crops from the user-provided false-positive camera scene. The negative set is small and based on one scene; results do not establish general false-positive performance.",
    })
    (run / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8"
    )

    target = ROOT / "custom_objects"
    target.mkdir(exist_ok=True)
    pending = target / "model.pending"
    shutil.copyfile(best, pending)
    pending.replace(target / "model.pt")
    manifest = {
        "classes": ["charger"],
        "sha256": hashlib.sha256((target / "model.pt").read_bytes()).hexdigest(),
        "source_run": run.relative_to(ROOT).as_posix(),
        "trained_by_user": True,
        "status": "experimental",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print("Charger model saved and activated:", target / "model.pt")
    print("Held-out test metrics:", run / "test_metrics.json")


if __name__ == "__main__":
    main()
