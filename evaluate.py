import os
from ultralytics import YOLO

def evaluate():
    # Gunakan bobot hasil training jika ada, atau fallback ke model default
    weights_path = "weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolov8n.pt"
        print(f"[INFO] Model custom belum ditemukan. Menggunakan model default: {weights_path}")
    else:
        print(f"[INFO] Evaluasi model custom: {weights_path}")

    model = YOLO(weights_path)

    config_path = "data.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] File '{config_path}' tidak ditemukan!")
        return

    print("=" * 60)
    print(" Evaluasi Performa Model YOLOv8 (mAP, Precision, Recall)")
    print("=" * 60)

    # Jalankan proses evaluasi/validasi
    metrics = model.val(data=config_path, split="val")

    print("\n--- HASIL METRIK EVALUASI ---")
    print(f"mAP@50     : {metrics.box.map50:.4f}")
    print(f"mAP@50-95  : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")
    print("--------------------------------")
    print(f"Grafik & Confusion Matrix tersimpan di: {metrics.save_dir}")

if __name__ == "__main__":
    evaluate()
