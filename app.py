"""
app.py — Streamlit Deepfake Detection App with IMAGE + VIDEO modes.
"""

import os, sys, tempfile, time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.main .block-container{padding-top:1.5rem;max-width:1200px}
.result-card{border-radius:16px;padding:2rem;text-align:center;border:2px solid rgba(255,255,255,.1)}
.result-fake{border-color:#FF4B4B55;background:linear-gradient(135deg,rgba(255,75,75,.08),rgba(255,75,75,.02))}
.result-real{border-color:#00C85355;background:linear-gradient(135deg,rgba(0,200,83,.08),rgba(0,200,83,.02))}
.result-label{font-size:3rem;font-weight:900;letter-spacing:.05em;margin-bottom:.5rem}
.label-fake{color:#FF4B4B}.label-real{color:#00C853}
.conf-text{font-size:1rem;color:#ccc}
.model-badge{display:inline-block;padding:.25rem .75rem;border-radius:20px;font-size:.8rem;font-weight:600;margin:.2rem;border:1px solid rgba(255,255,255,.1)}
.badge-ok{background:rgba(0,200,83,.15);color:#00C853;border-color:#00C85355}
.badge-miss{background:rgba(255,75,75,.15);color:#FF4B4B;border-color:#FF4B4B55}
.stProgress>div>div{background:linear-gradient(90deg,#6C63FF,#E040FB)!important;border-radius:4px}
</style>""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from src.inference.predictor import DeepfakeDetector
    from src.utils.config_loader import load_config
    from src.utils.helpers import get_device
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK, IMPORT_ERROR = False, str(e)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    try:
        return DeepfakeDetector()
    except Exception as e:
        return str(e)

@st.cache_resource(show_spinner=False)
def load_cfg():
    try:
        return load_config()
    except Exception:
        return {}


# ── Chart helpers ─────────────────────────────────────────────────────────────
def gauge(value: float, label: str, is_fake: bool) -> go.Figure:
    color = "#FF4B4B" if is_fake else "#00C853"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        title={"text": label, "font": {"size": 13, "color": "#ccc"}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": .3},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0,50], "color": "rgba(0,200,83,.08)"},
                          {"range": [50,100], "color": "rgba(255,75,75,.08)"}],
               "threshold": {"line": {"color": "#fff", "width": 2},
                              "thickness": .75, "value": 50}},
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=35,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
    return fig


def model_bars(per_model: Dict) -> go.Figure:
    names = [n.upper() for n in per_model]
    vals  = [per_model[n]["fake_probability"] * 100 for n in per_model]
    colors= ["#FF4B4B" if v >= 50 else "#00C853" for v in vals]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                            text=[f"{v:.1f}%" for v in vals], textposition="outside",
                            hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>"))
    fig.add_hline(y=50, line_dash="dash", line_color="#FFB300")
    fig.update_layout(height=240, yaxis=dict(range=[0,115]),
                      margin=dict(l=0,r=0,t=30,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ccc", showlegend=False,
                      title=dict(text="Per-Model Fake %", font=dict(size=13)))
    return fig


def timeline(probs: List[float]) -> go.Figure:
    x = list(range(1, len(probs)+1))
    colors = ["#FF4B4B" if p >= .5 else "#00C853" for p in probs]
    fig = go.Figure()
    fig.add_hline(y=.5, line_dash="dash", line_color="#FFB300", annotation_text="50%")
    fig.add_hrect(y0=0,y1=.5, fillcolor="rgba(0,200,83,.04)", line_width=0)
    fig.add_hrect(y0=.5,y1=1, fillcolor="rgba(255,75,75,.04)", line_width=0)
    fig.add_trace(go.Scatter(x=x, y=probs, mode="lines+markers",
                             line=dict(color="#6C63FF", width=2),
                             marker=dict(color=colors, size=7),
                             hovertemplate="Frame %{x}: <b>%{y:.1%}</b><extra></extra>"))
    fig.update_layout(height=240, xaxis_title="Frame", yaxis_title="Fake Prob",
                      yaxis=dict(range=[0,1], tickformat=".0%"),
                      margin=dict(l=0,r=0,t=30,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ccc",
                      title=dict(text="Frame-Level Analysis", font=dict(size=13)))
    return fig


def show_result_card(result: Dict):
    """Render the verdict card + metrics + charts."""
    is_fake = result["label_idx"] == 1
    cls     = "result-fake" if is_fake else "result-real"
    lc      = "label-fake" if is_fake else "label-real"
    icon    = "🚨" if is_fake else "✅"

    st.markdown(f"""
    <div class='result-card {cls}'>
      <div class='result-label {lc}'>{icon} {result["label"]}</div>
      <div class='conf-text'>
        Confidence: <b>{result["confidence"]:.1%}</b> &nbsp;|&nbsp;
        Frames: <b>{result.get("num_frames","—")}</b> &nbsp;|&nbsp;
        Models: <b>{result.get("num_models","—")}</b>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🎭 Fake Prob",  f"{result['fake_probability']:.1%}")
    c2.metric("✅ Real Prob",  f"{result['real_probability']:.1%}")
    c3.metric("⏱ Time",       result.get("processing_time_str","—"))
    c4.metric("🔀 Method",     result.get("method","—"))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gauge(result["fake_probability"], "Fake Probability", is_fake),
                        use_container_width=True)
    with col2:
        if result.get("per_model_probs"):
            st.plotly_chart(model_bars(result["per_model_probs"]), use_container_width=True)

    # Frame timeline only for video
    fp = result.get("frame_level_fake_probs", [])
    if len(fp) > 1:
        st.plotly_chart(timeline(fp), use_container_width=True)

    # Per-model breakdown
    if result.get("per_model_probs"):
        st.markdown("#### Per-Model Breakdown")
        cols = st.columns(len(result["per_model_probs"]))
        for col,(name,data) in zip(cols, result["per_model_probs"].items()):
            with col:
                fp2 = data["fake_probability"]
                pred = data["prediction"]
                dc = "inverse" if pred=="FAKE" else "normal"
                st.metric(f"**{name.upper()}**", pred, f"Fake: {fp2:.1%}",
                           delta_color=dc)


def show_gradcam(gradcam_maps: Dict, face_crop: Optional[np.ndarray]):
    """Render Grad-CAM heatmap section."""
    if not gradcam_maps:
        return
    st.markdown("#### 🔥 Grad-CAM Explanations")
    st.caption("Red = regions most influential for the prediction")

    cols = st.columns(len(gradcam_maps))
    for col, (name, gd) in zip(cols, gradcam_maps.items()):
        with col:
            st.markdown(f"**{name.upper()}**")
            # Show side-by-side: original | overlay
            row = st.columns(2)
            with row[0]:
                if face_crop is not None:
                    st.image(face_crop, caption="Face crop", use_column_width=True)
            with row[1]:
                overlay = gd.get("overlay")
                if overlay is not None:
                    st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)

            hmap = gd.get("heatmap")
            if hmap is not None:
                hmap_u8 = np.uint8(255 * hmap)
                colored  = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
                colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                st.image(colored_rgb, caption="Heatmap", use_column_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar(config: Dict, detector) -> Dict:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:.5rem 0 1rem'>
          <span style='font-size:2.5rem'>🔍</span>
          <h2 style='margin:0;color:#6C63FF'>DeepFake<br>Detector</h2>
          <p style='color:#888;font-size:.8rem'>ResNet50 · VGG16 · InceptionV3</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # Model status
        st.markdown("**🤖 Models**")
        for name in ["resnet50","vgg16","inceptionv3"]:
            if isinstance(detector, DeepfakeDetector):
                ok = detector.model_loaded.get(name, False)
                cls = "badge-ok" if ok else "badge-miss"
                icon = "✓" if ok else "✗"
            else:
                cls, icon = "badge-miss", "✗"
            st.markdown(f"<span class='model-badge {cls}'>{icon} {name.upper()}</span>",
                        unsafe_allow_html=True)
        st.markdown("---")

        fe  = config.get("frame_extraction", {})
        ens = config.get("ensemble", {})

        frame_interval = st.slider("Frame interval", 1, 30,
                                    fe.get("frame_interval", 10))
        max_frames     = st.slider("Max frames", 5, 100,
                                    fe.get("max_frames_per_video", 30))
        face_margin    = st.slider("Face margin", 0.0, 0.6,
                                    fe.get("face_margin", 0.3), 0.05)
        ensemble_method= st.selectbox("Ensemble method",
                                       ["soft_voting","weighted_avg","hard_voting"])
        threshold      = st.slider("Fake threshold", 0.3, 0.9,
                                    ens.get("confidence_threshold", 0.5), 0.05)
        show_gradcam_cb= st.checkbox("Show Grad-CAM (images)", value=True)

        with st.expander("💻 System"):
            import torch
            d = get_device("auto")
            st.write(f"PyTorch: {torch.__version__}")
            st.write(f"Device:  {d}")
            if d.type == "cuda":
                st.write(f"GPU: {torch.cuda.get_device_name(0)}")

    return dict(frame_interval=frame_interval, max_frames=max_frames,
                face_margin=face_margin, ensemble_method=ensemble_method,
                threshold=threshold, show_gradcam=show_gradcam_cb)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:1.5rem'>
      <h1 style='font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#6C63FF,#E040FB);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        🔍 DeepFake Detector</h1>
      <p style='color:#aaa'>Ensemble AI · ResNet50 + VGG16 + InceptionV3 · MediaPipe Face Detection</p>
    </div>""", unsafe_allow_html=True)

    if not IMPORTS_OK:
        st.error(f"Import error: {IMPORT_ERROR}")
        return

    config = load_cfg()
    with st.spinner("Loading models…"):
        detector = load_detector()

    detector_ready = isinstance(detector, DeepfakeDetector)
    if not detector_ready:
        st.warning(
            f"⚠️ Models not loaded: {detector}\n\n"
            "**Run training first:**\n"
            "```bash\n"
            "python src/data/dataset_downloader.py --dataset celebdf_v2\n"
            "python src/data/real_dataset_prep.py\n"
            "python src/training/train_pipeline.py\n"
            "```"
        )
    else:
        n = sum(detector.model_loaded.values())
        st.success(f"✅ {n}/3 models loaded and ready.")

    settings = sidebar(config, detector if detector_ready else None)

    tab_img, tab_vid, tab_demo, tab_about = st.tabs([
        "🖼️ Image Detection", "🎬 Video Detection", "📊 Demo", "ℹ️ About"
    ])

    # ═══════════════════════════════════════════════════════════
    # IMAGE TAB
    # ═══════════════════════════════════════════════════════════
    with tab_img:
        st.markdown("### Upload Image for Deepfake Analysis")
        col_up, col_guide = st.columns([2,1])

        with col_up:
            img_file = st.file_uploader(
                "Upload image", type=["jpg","jpeg","png","bmp","webp"],
                key="img_uploader", label_visibility="collapsed"
            )

        with col_guide:
            st.markdown("""
            **IMAGE MODE**
            - Supports JPG, PNG, BMP, WebP
            - Face is automatically detected
            - Grad-CAM shows *why* the model decided
            - All 3 models vote via ensemble
            """)

        if img_file:
            st.markdown("---")
            pil_img = Image.open(img_file).convert("RGB")
            col_prev, col_info = st.columns([1,2])
            with col_prev:
                st.image(pil_img, caption="Uploaded image", use_column_width=True)
            with col_info:
                w, h = pil_img.size
                st.markdown(f"**File:** {img_file.name}")
                st.markdown(f"**Size:** {w}×{h} px — {img_file.size/1024:.1f} KB")

            if st.button("🔍 Analyze Image", type="primary", disabled=not detector_ready,
                          use_container_width=True):
                pb = st.progress(0)
                status = st.empty()

                def prog(f, msg):
                    pb.progress(min(f, 1.0))
                    status.text(f"⏳ {msg}")

                # Apply sidebar settings
                detector.face_extractor.face_margin       = settings["face_margin"]
                detector.ensemble.confidence_threshold    = settings["threshold"]
                from src.inference.ensemble import EnsembleMethod
                try:
                    detector.ensemble.method = EnsembleMethod(settings["ensemble_method"])
                except Exception:
                    pass

                try:
                    img_array = np.array(pil_img)
                    result = detector.predict_image(
                        img_array,
                        return_gradcam=settings["show_gradcam"],
                        progress_callback=prog,
                    )
                    pb.progress(1.0); status.empty()

                    st.markdown("---")
                    st.markdown("## 🎯 Result")
                    show_result_card(result)

                    # Grad-CAM section
                    if settings["show_gradcam"] and result.get("gradcam_maps"):
                        st.markdown("---")
                        show_gradcam(result["gradcam_maps"], result.get("face_crop"))

                    if not result.get("face_detected", True):
                        st.info("ℹ️ No face detected — full image was analyzed.")

                except Exception as e:
                    pb.empty(); status.empty()
                    st.error(f"❌ Analysis failed: {e}")
                    with st.expander("Error details"):
                        import traceback
                        st.code(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════
    # VIDEO TAB
    # ═══════════════════════════════════════════════════════════
    with tab_vid:
        st.markdown("### Upload Video for Deepfake Analysis")
        col_up2, col_guide2 = st.columns([2,1])

        with col_up2:
            vid_file = st.file_uploader(
                "Upload video", type=["mp4","avi","mov","mkv","wmv","m4v"],
                key="vid_uploader", label_visibility="collapsed"
            )

        with col_guide2:
            st.markdown("""
            **VIDEO MODE**
            - Supports MP4, AVI, MOV, MKV
            - Frames extracted every N frames
            - Face detected per frame
            - Frame-level timeline shown
            - Ensemble aggregates all frames
            """)

        if vid_file:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(vid_file.name).suffix
            ) as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            st.markdown("---")
            col_v, col_prev = st.columns([1,1])
            with col_v:
                st.markdown("#### 🎥 Uploaded Video")
                st.video(tmp_path)
                sz_mb = Path(tmp_path).stat().st_size / 1e6
                st.caption(f"{vid_file.name} · {sz_mb:.1f} MB")

            with col_prev:
                st.markdown("#### Preview Frames")
                cap = cv2.VideoCapture(tmp_path)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prev_imgs = []
                for fi in [int(total_f*i/6) for i in range(6)]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, fr = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        pil.thumbnail((150,110))
                        prev_imgs.append(pil)
                cap.release()
                if prev_imgs:
                    pc = st.columns(3)
                    for i,im in enumerate(prev_imgs[:6]):
                        with pc[i%3]:
                            st.image(im, use_column_width=True)

            if st.button("🔍 Analyze Video", type="primary", disabled=not detector_ready,
                          use_container_width=True):
                pb2 = st.progress(0)
                st2 = st.empty()

                def prog2(f, msg):
                    pb2.progress(min(f, 1.0)); st2.text(f"⏳ {msg}")

                detector.face_extractor.frame_interval    = settings["frame_interval"]
                detector.face_extractor.max_frames        = settings["max_frames"]
                detector.face_extractor.face_margin       = settings["face_margin"]
                detector.ensemble.confidence_threshold    = settings["threshold"]

                try:
                    result = detector.predict_video(tmp_path, progress_callback=prog2)
                    pb2.progress(1.0); st2.empty()
                    st.markdown("---")
                    st.markdown("## 🎯 Result")
                    show_result_card(result)
                except Exception as e:
                    pb2.empty(); st2.empty()
                    st.error(f"❌ {e}")
                    with st.expander("Error details"):
                        import traceback
                        st.code(traceback.format_exc())
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    # ═══════════════════════════════════════════════════════════
    # DEMO TAB
    # ═══════════════════════════════════════════════════════════
    with tab_demo:
        st.markdown("### 📊 Demo Mode — synthetic results (no models needed)")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚨 Show FAKE Example", type="primary", use_container_width=True):
                st.session_state["demo"] = {
                    "label":"FAKE","label_idx":1,"confidence":.87,
                    "fake_probability":.87,"real_probability":.13,
                    "num_frames":24,"num_models":3,"processing_time_str":"8.4s",
                    "method":"soft_voting",
                    "frame_level_fake_probs":[.55,.72,.81,.79,.88,.92,.85,.90,.87,.83,
                                              .78,.91,.93,.86,.88,.95,.84,.89,.82,.91,
                                              .93,.87,.90,.88],
                    "per_model_probs":{
                        "resnet50":   {"fake_probability":.84,"real_probability":.16,"prediction":"FAKE"},
                        "vgg16":      {"fake_probability":.89,"real_probability":.11,"prediction":"FAKE"},
                        "inceptionv3":{"fake_probability":.88,"real_probability":.12,"prediction":"FAKE"},
                    },
                }
        with c2:
            if st.button("✅ Show REAL Example", use_container_width=True):
                st.session_state["demo"] = {
                    "label":"REAL","label_idx":0,"confidence":.91,
                    "fake_probability":.09,"real_probability":.91,
                    "num_frames":20,"num_models":3,"processing_time_str":"7.1s",
                    "method":"soft_voting",
                    "frame_level_fake_probs":[.12,.08,.15,.07,.10,.09,.11,.08,.13,.06,
                                              .09,.10,.07,.11,.08,.09,.12,.07,.10,.09],
                    "per_model_probs":{
                        "resnet50":   {"fake_probability":.08,"real_probability":.92,"prediction":"REAL"},
                        "vgg16":      {"fake_probability":.10,"real_probability":.90,"prediction":"REAL"},
                        "inceptionv3":{"fake_probability":.09,"real_probability":.91,"prediction":"REAL"},
                    },
                }

        if "demo" in st.session_state:
            st.markdown("---")
            show_result_card(st.session_state["demo"])

    # ═══════════════════════════════════════════════════════════
    # ABOUT TAB
    # ═══════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("""
## About This System

**Ensemble AI** using ResNet50 + VGG16 + InceptionV3 with MediaPipe face detection.

### Supported Datasets
| Dataset | Type | Size | Access |
|---------|------|------|--------|
| **Celeb-DF v2** | Video | 590R + 5639F | Public (Google Drive) |
| **FaceForensics++** | Video | ~1000 each | Request form |
| **DFDC** | Video | ~100k | Kaggle |
| **Custom** | Video/Image | Any | Your own |

### Pipeline
```
Upload (Image or Video)
  → Face Detection (MediaPipe)
  → Preprocessing (resize, normalize)
  → ResNet50  (224×224)
  → VGG16     (224×224)
  → InceptionV3 (299×299)
  → Soft Voting Ensemble
  → REAL / FAKE + Confidence
  → Grad-CAM Explanation (image mode)
```

### Quick Start
```bash
# 1. Download dataset
python src/data/dataset_downloader.py --dataset celebdf_v2

# 2. Preprocess
python src/data/real_dataset_prep.py

# 3. Train
python src/training/train_pipeline.py

# 4. Launch UI
streamlit run app.py
```
        """)


if __name__ == "__main__":
    main()
