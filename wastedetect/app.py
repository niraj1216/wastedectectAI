"""WasteDetect AI - drag & drop street photos, get litter severity + report. Low-RAM (ONNX Runtime).
Run: streamlit run app.py"""
import glob, os
import cv2, pandas as pd
import streamlit as st
import waste_core as wc

st.set_page_config(page_title="WasteDetect AI", page_icon="♻️", layout="wide")
st.title("♻️ WasteDetect AI")
st.caption("Drag & drop street photos (even random ones from Google). Get detections, a litter-severity score and a cleanup report.")
MAX_FILES = 5

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.8, 0.20, 0.05,
                     help="Lower = finds more litter but more false positives.")
    st.markdown("**Score** = 60% weighted item count + 40% image area covered by detections.")
    st.caption(f"Up to {MAX_FILES} images at once; large photos are downscaled to {wc.MAX_SIDE}px.")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return wc.load_model()


try:
    get_model()
except Exception as e:
    st.error(f"Model not available: {e}")
    st.stop()

files = st.file_uploader("Drag & drop one or more images here", type=["jpg", "jpeg", "png", "webp"],
                         accept_multiple_files=True)
if files and len(files) > MAX_FILES:
    st.warning(f"Only the first {MAX_FILES} images are analyzed (memory limit).")
    files = files[:MAX_FILES]

items = []
for f in files or []:
    img = wc.load_image(f.getvalue())
    if img is not None:
        items.append((f.name, img))
samples = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*.[jp][pn]g")))
if not items and samples:
    pick = st.selectbox("...or try a sample", ["-"] + samples,
                        format_func=lambda p: p if p == "-" else os.path.basename(p))
    if pick != "-":
        with open(pick, "rb") as fh:
            items.append((os.path.basename(pick), wc.load_image(fh.read())))

if not items:
    st.info("Drop an image above to begin.")
    st.stop()

results = []
for name, img in items:
    with st.spinner(f"Analyzing {name}..."):
        annotated, a = wc.analyze(img, conf)
    results.append((name, img, annotated, a))

if len(results) > 1:
    st.subheader("Batch summary (worst first)")
    df = pd.DataFrame([{"Image": n, "Score": a.score, "Priority": a.level, "Items": len(a.detections),
                        "Coverage %": round(a.coverage * 100)} for n, _, _, a in results]).sort_values("Score", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download summary CSV", df.to_csv(index=False), "wastedetect_summary.csv", "text/csv")

for name, img, annotated, a in results:
    st.divider()
    st.header(name)
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Items detected", len(a.detections))
    m2.metric("Waste categories", len(a.counts))
    m3.metric("Area covered", f"{a.coverage:.0%}")
    m4.metric("Cleanup priority", a.level)

    g, b = st.columns(2)
    g.markdown(
        f"<div style='font-size:14px;color:#888'>Litter severity (0-100)</div>"
        f"<div style='font-size:56px;font-weight:800;color:{a.color};line-height:1.1'>{a.score:.0f}"
        f"<span style='font-size:22px;color:#888'>/100</span></div>"
        f"<div style='background:#8884;border-radius:12px;height:26px;margin-top:8px'>"
        f"<div style='width:{min(a.score,100)}%;background:{a.color};height:26px;border-radius:12px'></div></div>"
        f"<div style='display:flex;justify-content:space-between;font-size:12px;color:#888'>"
        f"<span>Low</span><span>Moderate</span><span>High</span><span>Critical</span></div>",
        unsafe_allow_html=True)
    if a.counts:
        b.caption("Detected items by type")
        b.bar_chart(pd.Series(a.counts, name="Count"), horizontal=True)
    else:
        b.success("No litter detected at this threshold. Try lowering the confidence slider.")

    st.subheader(f"Recommended next steps: {a.level} priority")
    for step in a.actions:
        st.markdown(f"- {step}")
    st.download_button("Download report (HTML, print to PDF)",
                       wc.build_report_html(name, img, annotated, a),
                       f"wastedetect_report_{os.path.splitext(name)[0]}.html", "text/html", key=f"dl-{name}")
