import os
import time
import cv2
from ultralytics import YOLO

# Path Model Terlatih
MODEL_CUSTOM_PATH_1 = os.path.join("weights", "best.pt")
MODEL_CUSTOM_PATH_2 = os.path.join("weight", "best.pt")
MODEL_DEFAULT_PATH = "yolov8n.pt"

CLASS_COLORS = {
    0: (0, 0, 255),     # Pothole: Merah (BGR)
    1: (0, 215, 255),   # Crack: Kuning Emas (BGR)
}
DEFAULT_COLOR = (0, 255, 0)
CLASS_NAMES = {0: "Pothole", 1: "Crack"}

def draw_header_overlay(frame, counts, fps):
    """
    Menggambar panel informasi statistik di pojok kiri atas video stream.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    panel_w = 340
    panel_h = 120
    
    # Latar belakang semi-transparan
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 255, 255), 1)
    
    # Judul
    cv2.putText(frame, "REAL-TIME ROAD DAMAGE DETECTION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Statistik Deteksi
    pothole_cnt = counts.get(0, 0)
    crack_cnt = counts.get(1, 0)
    total_cnt = pothole_cnt + crack_cnt
    
    cv2.putText(frame, f"Pothole: {pothole_cnt} | Crack: {crack_cnt}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
    cv2.putText(frame, f"Total Kerusakan: {total_cnt}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"Kecepatan: {fps:.1f} FPS | [S]: Save | [Q]: Quit", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def run_webcam():
    # 1. Load Model YOLOv8
    if os.path.exists(MODEL_CUSTOM_PATH_1):
        model_path = MODEL_CUSTOM_PATH_1
    elif os.path.exists(MODEL_CUSTOM_PATH_2):
        model_path = MODEL_CUSTOM_PATH_2
    else:
        model_path = MODEL_DEFAULT_PATH

    print(f"[INFO] Memuat Model: {model_path}")
    model = YOLO(model_path)

    # 2. Inisialisasi Kamera Live (0 = Kamera Bawaan Laptop/USB Dashcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak ditemukan! Pastikan webcam/kamera terhubung.")
        return

    # Atur resolusi kamera ke HD 720p jika didukung
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    os.makedirs("output", exist_ok=True)
    snapshot_count = 0

    print("=" * 65)
    print(" APLIKASI REAL-TIME DETEKSI KERUSAKAN JALAN BERJALAN")
    print(" Tekan tombol 'S' di jendela kamera untuk mengambil foto snapshot.")
    print(" Tekan tombol 'Q' untuk keluar dari aplikasi.")
    print("=" * 65)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Gagal membaca frame kamera.")
            break

        # Hitung FPS real-time
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        # Run Inferensi YOLOv8 pada frame live (Confidence threshold = 0.4)
        results = model.predict(frame, conf=0.4, verbose=False)[0]

        counts = {0: 0, 1: 0}
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                counts[cls_id] = counts.get(cls_id, 0) + 1
                color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                label_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")

                # Bounding Box & Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                caption = f"{label_name} {conf:.2f}"
                (w_txt, h_txt), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + w_txt + 6, max(0, y1)), color, -1)
                cv2.putText(frame, caption, (x1 + 3, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # Draw Header Panel
        draw_header_overlay(frame, counts, fps)

        # Tampilkan Jendela Camera Feed
        cv2.imshow("Real-Time Road Damage Detection (YOLOv8)", frame)

        # Tombol Pintas Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            snapshot_count += 1
            snap_path = os.path.join("output", f"snapshot_realtime_{snapshot_count}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[BERHASIL] Tangkapan layar disimpan: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Aplikasi real-time kamera dihentikan.")

if __name__ == "__main__":
    run_webcam()
