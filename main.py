import os
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Direktori Konfigurasi
MODEL_CUSTOM_PATH_1 = os.path.join("weights", "best.pt")
MODEL_CUSTOM_PATH_2 = os.path.join("weight", "best.pt")
MODEL_DEFAULT_PATH = "yolov8n.pt"

INPUT_IMAGE_FOLDER = "input_image"
INPUT_VIDEO_FOLDER = "input_video"
OUTPUT_FOLDER = "output"

# Warna Bounding Box per Kelas (Format BGR)
CLASS_COLORS = {
    0: (0, 0, 255),     # Pothole: Merah (Red)
    1: (0, 215, 255),   # Crack: Kuning Emas (Gold)
}
DEFAULT_COLOR = (0, 255, 0) # Hijau jika ada kelas tak dikenal

CLASS_NAMES_DEFAULT = {0: "Pothole", 1: "Crack"}

def draw_header_overlay(frame, counts, class_names, fps=None):
    """
    Menggambar panel overlay informasi rekapitulasi deteksi di sudut kiri atas frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # Tinggi panel disesuaikan dengan jumlah informasi
    panel_h = 100 if fps is None else 125
    panel_w = min(360, w - 20)
    
    # Latar belakang transparan gelap (Glassmorphism effect)
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Border panel
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (200, 200, 200), 1)
    
    # Judul Panel
    cv2.putText(frame, "DETEKSI KERUSAKAN JALAN", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    
    # Teks Jumlah Deteksi per Kelas
    y_offset = 60
    total_damage = sum(counts.values())
    
    info_str = " | ".join([f"{class_names.get(cid, f'Class {cid}')}: {cnt}" for cid, cnt in counts.items()])
    cv2.putText(frame, info_str, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Total Kerusakan: {total_damage}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    if fps is not None:
        y_offset += 25
        cv2.putText(frame, f"Kecepatan: {fps:.1f} FPS", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

def process_image(image_path, model, output_folder, summary_report):
    """
    Memproses file citra tunggal, menggambar deteksi, dan menyimpannya.
    """
    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Gagal membaca gambar: {image_path}")
        return

    results = model.predict(image_path, conf=0.4, save=False)[0]
    
    counts = {0: 0, 1: 0} # 0: Pothole, 1: Crack
    boxes = results.boxes

    class_names = results.names if hasattr(results, 'names') else CLASS_NAMES_DEFAULT

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            counts[cls_id] = counts.get(cls_id, 0) + 1
            color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
            label_name = class_names.get(cls_id, f"Class_{cls_id}")
            
            # Gambar Bounding Box & Label
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            caption = f"{label_name} {conf:.2f}"
            
            # Background teks label
            (w_txt, h_txt), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - 20)), (x1 + w_txt + 4, max(0, y1)), color, -1)
            cv2.putText(img, caption, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Tambahkan Header Summary Overlay
    draw_header_overlay(img, counts, class_names)

    # Simpan Hasil
    output_path = os.path.join(output_folder, f"deteksi_{image_name}")
    cv2.imwrite(output_path, img)
    print(f"[BERHASIL] Gambar diproses: {image_name} -> {output_path}")

    # Rekap data
    summary_report["images"][image_name] = {
        "pothole": counts.get(0, 0),
        "crack": counts.get(1, 0),
        "total": sum(counts.values()),
        "output_file": output_path
    }

def process_video(video_path, model, output_folder, summary_report):
    """
    Memproses file video, melakukan inferensi frame-by-frame, dan menyimpan video output.
    """
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Gagal membuka video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_path = os.path.join(output_folder, f"deteksi_{video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    frame_count = 0
    total_counts = {0: 0, 1: 0}

    print(f"[INFO] Memproses video: {video_name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        results = model.predict(frame, conf=0.4, verbose=False)[0]
        proc_time = time.time() - start_time
        fps_curr = 1.0 / proc_time if proc_time > 0 else 0

        frame_counts = {0: 0, 1: 0}
        boxes = results.boxes
        class_names = results.names if hasattr(results, 'names') else CLASS_NAMES_DEFAULT

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                frame_counts[cls_id] = frame_counts.get(cls_id, 0) + 1
                color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                label_name = class_names.get(cls_id, f"Class_{cls_id}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                caption = f"{label_name} {conf:.2f}"
                (w_txt, h_txt), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + w_txt + 4, max(0, y1)), color, -1)
                cv2.putText(frame, caption, (x1 + 2, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        draw_header_overlay(frame, frame_counts, class_names, fps=fps_curr)
        out.write(frame)

        for k, v in frame_counts.items():
            total_counts[k] += v
        frame_count += 1

    cap.release()
    out.release()
    print(f"[BERHASIL] Video diproses ({frame_count} frames): {video_name} -> {output_path}")

    summary_report["videos"][video_name] = {
        "frames_processed": frame_count,
        "output_file": output_path
    }

def main():
    # 1. Menentukan Model YOLOv8
    if os.path.exists(MODEL_CUSTOM_PATH_1):
        model_path = MODEL_CUSTOM_PATH_1
        print(f"[INFO] Menggunakan model custom terlatih: {model_path}")
    elif os.path.exists(MODEL_CUSTOM_PATH_2):
        model_path = MODEL_CUSTOM_PATH_2
        print(f"[INFO] Menggunakan model custom terlatih: {model_path}")
    else:
        model_path = MODEL_DEFAULT_PATH
        print(f"[INFO] Model custom 'best.pt' belum ditemukan. Fallback ke model dasar: {model_path}")

    model = YOLO(model_path)

    # 2. Siapkan Folder Input/Output
    os.makedirs(INPUT_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(INPUT_VIDEO_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    summary_report = {"images": {}, "videos": {}}

    # 3. Proses File Citra (Gambar)
    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        print(f"\n--- Memproses {len(image_files)} Gambar di '{INPUT_IMAGE_FOLDER}/' ---")
        for img_file in image_files:
            process_image(os.path.join(INPUT_IMAGE_FOLDER, img_file), model, OUTPUT_FOLDER, summary_report)
    else:
        print(f"[INFO] Tidak ada berkas gambar di '{INPUT_IMAGE_FOLDER}/'. Silakan masukkan gambar uji.")

    # 4. Proses File Video
    video_files = [f for f in os.listdir(INPUT_VIDEO_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if video_files:
        print(f"\n--- Memproses {len(video_files)} Video di '{INPUT_VIDEO_FOLDER}/' ---")
        for vid_file in video_files:
            process_video(os.path.join(INPUT_VIDEO_FOLDER, vid_file), model, OUTPUT_FOLDER, summary_report)
    else:
        print(f"[INFO] Tidak ada berkas video di '{INPUT_VIDEO_FOLDER}/'. Silakan masukkan video uji.")

    # 5. Simpan Laporan Rekapitulasi Deteksi (JSON)
    report_path = os.path.join(OUTPUT_FOLDER, "rekapitulasi_deteksi.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    print("\n" + "=" * 60)
    print(" SELURUH PROSES DETEKSI SELESAI")
    print(f" Berkas output & Laporan Rekapitulasi tersimpan di: {os.path.abspath(OUTPUT_FOLDER)}")
    print("=" * 60)

if __name__ == "__main__":
    main()