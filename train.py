import os
import shutil
from ultralytics import YOLO

def main():
    # 1. Pastikan file data.yaml tersedia
    config_path = "data.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] File konfigurasi '{config_path}' tidak ditemukan!")
        print("Pastikan data.yaml berada di direktori proyek ini.")
        return

    # 2. Inisialisasi model pre-trained YOLOv8
    # Varian yang didukung: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model_version = "yolov8n.pt"
    print(f"[INFO] Menguji/Melatih model YOLOv8 berbasis: {model_version}")
    model = YOLO(model_version)

    # 3. Parameter Pelatihan (Hyperparameters)
    epochs = 50
    imgsz = 640
    batch_size = 16
    project_name = "runs/detect"
    experiment_name = "road_damage_model"

    print("=" * 60)
    print(f" Memulai Pelatihan YOLOv8 - Kerusakan Jalan (Pothole & Crack)")
    print(f" Epochs: {epochs} | Image Size: {imgsz} | Batch Size: {batch_size}")
    print("=" * 60)

    # 4. Jalankan Pelatihan
    try:
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name=experiment_name,
            project=project_name,
            exist_ok=True,
            save=True,
            plots=True
        )

        # 5. Salin bobot terbaik (best.pt) ke direktori weights/
        os.makedirs("weights", exist_ok=True)
        trained_best_weights = os.path.join(project_name, experiment_name, "weights", "best.pt")
        target_weights = os.path.join("weights", "best.pt")

        if os.path.exists(trained_best_weights):
            shutil.copy(trained_best_weights, target_weights)
            print(f"\n[BERHASIL] Model terbaik disimpan ke: {os.path.abspath(target_weights)}")
        else:
            print("\n[CATATAN] Pelatihan selesai. Cek folder runs/detect/ untuk bobot model.")

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat pelatihan: {e}")
        print("Pastikan folder dataset/ sudah diisi dengan data training & validation.")

if __name__ == "__main__":
    main()
