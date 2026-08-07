import os
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Coba import streamlit-webrtc untuk dukungan kamera browser di Cloud (HP & Laptop)
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Deteksi Kerusakan Jalan YOLOv8",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model Paths
MODEL_CUSTOM_PATH_1 = os.path.join("weights", "best.pt")
MODEL_CUSTOM_PATH_2 = os.path.join("weight", "best.pt")
MODEL_DEFAULT_PATH = "yolov8n.pt"

CLASS_NAMES = {0: "Pothole", 1: "Crack"}

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_CUSTOM_PATH_1):
        path = MODEL_CUSTOM_PATH_1
    elif os.path.exists(MODEL_CUSTOM_PATH_2):
        path = MODEL_CUSTOM_PATH_2
    else:
        path = MODEL_DEFAULT_PATH
    return YOLO(path), path

def main():
    st.title("🛣️ Sistem Real-Time Deteksi & Klasifikasi Kerusakan Jalan")
    st.caption("Implementasi Computer Vision Berbasis Algoritma YOLOv8 (Pothole & Crack Detection)")

    model, model_path = load_model()

    # Sidebar Controls
    st.sidebar.header("⚙️ Pengaturan & Kontrol")
    st.sidebar.success(f"Model Aktif: `{os.path.basename(model_path)}`")
    
    conf_threshold = st.sidebar.slider("Ambang Keyakinan (Confidence Threshold)", 0.1, 1.0, 0.35, 0.05)
    
    app_mode = st.sidebar.radio(
        "Pilih Mode Operasi:",
        ["📸 Deteksi Gambar (Image)", "🎥 Deteksi Video", "📹 Live Stream Kamera", "📊 Laporan Analitik"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Petunjuk Skripsi:** Gunakan mode **Live Stream Kamera** atau **Deteksi Video** saat demonstrasi pengujian real-time di hadapan dosen penguji.")

    # ---------------------------------------------------------
    # MODE 1: DETEKSI GAMBAR
    # ---------------------------------------------------------
    if app_mode == "📸 Deteksi Gambar (Image)":
        st.subheader("📸 Mode Deteksi Citra Gambar")
        uploaded_file = st.file_uploader("Unggah foto jalanan (.jpg, .jpeg, .png)", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar Original", use_container_width=True)

            with st.spinner("Mendeteksi kerusakan jalan..."):
                results = model.predict(img_array, conf=conf_threshold)[0]
                res_plotted = results.plot()

                counts = {0: 0, 1: 0}
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        counts[cls_id] = counts.get(cls_id, 0) + 1

            with col2:
                st.image(res_plotted, caption="Hasil Deteksi (Bounding Box & Label)", use_container_width=True)

            st.markdown("### 📊 Rekapitulasi Deteksi Citra")
            m1, m2, m3 = st.columns(3)
            m1.metric("Lubang (Pothole)", counts.get(0, 0))
            m2.metric("Retakan (Crack)", counts.get(1, 0))
            m3.metric("Total Kerusakan", sum(counts.values()))

    # ---------------------------------------------------------
    # MODE 2: DETEKSI VIDEO
    # ---------------------------------------------------------
    elif app_mode == "🎥 Deteksi Video":
        st.subheader("🎥 Mode Deteksi Rekaman Video")
        uploaded_video = st.file_uploader("Unggah video kondisi jalanan (.mp4, .avi, .mov)", type=['mp4', 'avi', 'mov'])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            
            st.info("Memulai pemutaran dan inferensi video real-time...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
                annotated_frame = results.plot()

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_frame_rgb, caption="Video Real-Time Detection Feed", use_container_width=True)

            cap.release()
            st.success("Pemrosesan video selesai!")

    # ---------------------------------------------------------
    # MODE 3: LIVE STREAM KAMERA (OPTIMIZED HP & LAPTOP BROWSER)
    # ---------------------------------------------------------
    elif app_mode == "📹 Live Stream Kamera":
        st.subheader("📹 Live Streaming Kamera Real-Time (HP & Laptop)")

        if HAS_WEBRTC:
            st.markdown("""
            **Petunjuk Kamera HP / Laptop:**
            1. Izinkan akses kamera pada browser Chrome/Safari kamu.
            2. Di HP, kamu bisa memilih **Kamera Belakang (Environment)** untuk mengarah ke jalan.
            3. Klik tombol **START** di bawah untuk memulai pemindaian.
            """)

            cam_facing = st.radio("Pilih Kamera HP:", ["Kamera Belakang (Jalanan)", "Kamera Depan"], horizontal=True)
            facing_mode = "environment" if "Belakang" in cam_facing else "user"

            class YOLOVideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.conf = conf_threshold
                    self.frame_count = 0
                    self.latest_annotated = None

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img_bgr = frame.to_ndarray(format="bgr24")

                    # Optimasi FPS & Latensi HP: jalankan inferensi YOLOv8 secara efisien
                    self.frame_count += 1
                    if self.frame_count % 2 == 0 or self.latest_annotated is None:
                        results = model.predict(img_bgr, conf=conf_threshold, verbose=False)[0]
                        self.latest_annotated = results.plot()

                    return av.VideoFrame.from_ndarray(self.latest_annotated, format="bgr24")

            webrtc_streamer(
                key="yolo-road-detection-mobile",
                video_processor_factory=YOLOVideoProcessor,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]}
                    ]
                }),
                media_stream_constraints={
                    "video": {"facingMode": facing_mode},
                    "audio": False
                }
            )
        else:
            st.write("Mode Kamera Lokal OpenCV:")
            run_cam = st.checkbox("Aktifkan Umpan Kamera (Live Camera Feed)")
            st_cam_frame = st.empty()

            if run_cam:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Gagal membuka kamera. Pastikan akses kamera diizinkan.")
                else:
                    prev_t = time.time()
                    while run_cam:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        curr_t = time.time()
                        fps = 1.0 / (curr_t - prev_t) if (curr_t - prev_t) > 0 else 30.0
                        prev_t = curr_t

                        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
                        annotated_frame = results.plot()
                        
                        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st_cam_frame.image(annotated_rgb, channels="RGB", use_container_width=True)

                    cap.release()

    # ---------------------------------------------------------
    # MODE 4: LAPORAN ANALITIK
    # ---------------------------------------------------------
    elif app_mode == "📊 Laporan Analitik":
        st.subheader("📊 Laporan Ringkasan Hasil Deteksi Skripsi")
        
        report_json = os.path.join("output", "rekapitulasi_deteksi.json")
        if os.path.exists(report_json):
            st.json(report_json)
        else:
            st.warning("Laporan JSON belum ada. Jalankan pengujian di main.py untuk memperbarui laporan.")

if __name__ == "__main__":
    main()
