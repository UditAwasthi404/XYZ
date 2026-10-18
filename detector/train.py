"""Fine-tune YOLOv8n on SKU-110K (dense shelf products).

Usage:
    python -m detector.train --epochs 50 --imgsz 1024 --batch 8
    # quick smoke test on Ultralytics' bundled dataset name:
    python -m detector.train --data SKU-110K.yaml --epochs 1 --fraction 0.01

Using the built-in name "SKU-110K.yaml" lets Ultralytics download (~11 GB)
and convert the dataset automatically. detector/sku110k.yaml documents the
layout and how to add 'gap' / 'price_tag' classes. Put the resulting
best.pt path in SHELFSENSE_WEIGHTS to use it in the API.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="SKU-110K.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set")
    ap.add_argument("--device", default=None, help="'cpu', '0', ...")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                fraction=a.fraction, device=a.device, project="runs/shelfsense", name="sku110k",
                close_mosaic=10, max_det=1000)
    print("Done. Export with: yolo export model=runs/shelfsense/sku110k/weights/best.pt format=onnx")


if __name__ == "__main__":
    main()
