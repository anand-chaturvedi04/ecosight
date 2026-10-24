"""EcoSight AI - Streamlit app. Run: streamlit run app.py"""
from __future__ import annotations

import html
import json
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.ui_theme import (background_html, bin_cards, detection_card, empty_state_html, footer_html,
                          friendly_name, hero_html, how_it_works_html, section_title, theme_css)
from src.waste_logic import (BINS, CATEGORIES, CATEGORY_ICONS, CLASS_TO_CATEGORY, CONF_CONFIDENT,
                             CONF_MIN, GENERAL, ORGANIC, RECYCLABLE, category_for, confidence_label,
                             get_impact_summary, nearest_bins, summarize)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
FALLBACK_MODEL_PATH = ROOT / "yolov8n.pt"
HISTORY_PATH = ROOT / "results" / "history.csv"
METRICS_PATH = ROOT / "results" / "metrics.json"

CATEGORY_LABELS = {ORGANIC: "Organic", RECYCLABLE: "Recyclable", GENERAL: "General / Non-recyclable"}
BIN_CSS = {ORGANIC: "bin-organic", RECYCLABLE: "bin-recycle", GENERAL: "bin-general"}

st.set_page_config(page_title="EcoSight AI", page_icon="🌿", layout="wide")


def keyed_container(key: str):
    """Container with a stable CSS class (st-key-<key>); falls back to a plain container on old Streamlit."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


@st.cache_resource(show_spinner="Loading vision model...")
def load_model():
    from ultralytics import YOLO
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH)), "custom"
    elif FALLBACK_MODEL_PATH.exists():
        return YOLO(str(FALLBACK_MODEL_PATH)), "base"
    return YOLO("yolov8n.pt"), "base"


def run_detection(model_tuple, image: Image.Image, conf: float, iou: float):
    model, mode = model_tuple
    bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])
    result = model.predict(bgr, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    annotated = np.ascontiguousarray(result.plot()[:, :, ::-1])
    dets = []
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        c = float(box.conf[0])
        mapped_cat = category_for(name)
        dets.append({
            "Item": name,
            "Category": mapped_cat or GENERAL,
            "Confidence": round(c, 3),
            "Certainty": confidence_label(c)
        })
    dets.sort(key=lambda d: -d["Confidence"])
    return annotated, dets


def save_history(filename: str, dets: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    aid, ts = uuid.uuid4().hex[:8], datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"analysis_id": aid, "timestamp": ts, "filename": filename, "item": d["Item"],
             "category": d["Category"], "confidence": d["Confidence"], "certainty": d["Certainty"]}
            for d in dets] or [{"analysis_id": aid, "timestamp": ts, "filename": filename, "item": "",
                                "category": "", "confidence": "", "certainty": ""}]
    df = pd.DataFrame(rows)
    df.to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)


def render_bins(active: str | None = None, counts: dict | None = None) -> None:
    cards = [{"css": BIN_CSS[cat], "icon": CATEGORY_ICONS[cat], "title": CATEGORY_LABELS[cat],
              "items": [friendly_name(c) for c, k in CLASS_TO_CATEGORY.items() if k == cat],
              "active": cat == active, "count": (counts or {}).get(cat, 0)} for cat in CATEGORIES]
    st.markdown(bin_cards(cards), unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌿 EcoSight AI")
st.sidebar.caption("Computer-vision waste segregation for campus waste management")

has_custom = MODEL_PATH.exists()
has_base = FALLBACK_MODEL_PATH.exists()
model_ready = has_custom or has_base

if has_custom:
    st.sidebar.success("Trained weights active: `models/best.pt` (11 Waste Classes)")
elif has_base:
    st.sidebar.info("Base detector active: `yolov8n.pt` (Demo Mode)")
else:
    st.sidebar.error("Model weights not found. Please train or place `yolov8n.pt`.")

conf_thr = st.sidebar.slider("Minimum confidence", 0.10, 0.90, CONF_MIN, 0.05)
iou_thr = st.sidebar.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
st.sidebar.info(f"Detections below {CONF_CONFIDENT:.0%} confidence are flagged **Uncertain**.")
motion = st.sidebar.toggle("Smooth animations", value=True)

# ------------------------------------------------------------------ hero
st.markdown(theme_css(motion) + background_html() + hero_html(), unsafe_allow_html=True)

tab_detect, tab_bins, tab_hist, tab_about = st.tabs(
    ["🌿 Detect & Sort", "🗺️ Smart Bin Finder", "📖 Field Journal", "🍃 About & System"])

# ------------------------------------------------------------------ detect
with tab_detect:
    if not model_ready:
        st.error("No model available. Place `yolov8n.pt` or run `python train.py --mode full`.")
    else:
        st.markdown(section_title("Show us what you found",
                                  "Capture a live snapshot or upload a photo to identify items and their target bin."),
                    unsafe_allow_html=True)

        input_choice = st.radio("Input Source:", ["📁 Upload Photo", "📸 Live Camera Snapshot"], horizontal=True)
        img_source = None
        source_name = "snapshot.jpg"

        if input_choice == "📁 Upload Photo":
            up = st.file_uploader("Upload a photo of waste items", type=["jpg", "jpeg", "png"])
            if up is not None:
                img_source = Image.open(up)
                source_name = up.name
        else:
            cam = st.camera_input("Capture waste item using camera")
            if cam is not None:
                img_source = Image.open(cam)
                source_name = f"cam_{datetime.now().strftime('%H%M%S')}.jpg"

        if img_source is not None:
            if st.button("🔍 Analyze Waste Item", type="primary", use_container_width=True):
                with st.spinner("Classifying materials & computing spatial contours..."):
                    loaded = load_model()
                    annotated, dets = run_detection(loaded, img_source, conf_thr, iou_thr)
                    save_history(source_name, dets)
                    st.session_state["last"] = {
                        "annotated": annotated,
                        "original": img_source,
                        "dets": dets,
                        "summary": summarize(dets),
                        "impact": get_impact_summary(dets)
                    }

        last = st.session_state.get("last")
        if last:
            c1, c2 = st.columns(2)
            c1.image(last["original"], caption="Original Input", use_container_width=True)
            c2.image(last["annotated"], caption="YOLOv8 Contours & Classes", use_container_width=True)

            s = last["summary"]
            icon = CATEGORY_ICONS.get(s["category"], "❔")
            st.markdown(f'<div class="verdict v-{s["status"]}">{icon} {html.escape(s["message"])}</div>',
                        unsafe_allow_html=True)

            # Virtual Smart Bin Actuator Simulation
            target_cat = s["category"]
            if target_cat:
                act_col1, act_col2 = st.columns([1, 2])
                with act_col1:
                    st.markdown("#### 🤖 Virtual Bin Actuator")
                    bin_color = "#2d6a4f" if target_cat == ORGANIC else ("#0284c7" if target_cat == RECYCLABLE else "#475569")
                    st.markdown(
                        f"""
                        <div style="padding:16px; border-radius:12px; background:{bin_color}; color:#ffffff; text-align:center;">
                            <div style="font-size:1.8rem; margin-bottom:4px;">{icon}</div>
                            <div style="font-weight:700; font-size:1.1rem;">LID OPEN (90°)</div>
                            <div style="font-size:0.85rem; opacity:0.9;">Target: {target_cat}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with act_col2:
                    st.markdown("#### ⚡ IoT Hardware Bridge")
                    serial_cmd = f"SERVO:{target_cat.split('/')[0].upper()}:OPEN:90"
                    st.code(serial_cmd, language="bash")
                    st.caption("Signals ready to transmit over USB/Serial or MQTT to smart bin microcontroller.")

            # Eco-Impact & Carbon Savings
            impact = last.get("impact")
            if impact and last["dets"]:
                st.markdown("#### 🌱 Eco-Impact & Contamination Advisory")
                e1, e2, e3 = st.columns(3)
                e1.metric("CO₂ Emissions Averted", f"{impact['total_co2_saved_g']} g CO₂e")
                e2.metric("Items Sorted", len(last["dets"]))
                primary_item = impact["items"][0] if impact["items"] else None
                e3.metric("Degradation Time", primary_item["degrade_time"] if primary_item else "N/A")

                for itm in impact["items"]:
                    with st.expander(f"Disposal Guide: {friendly_name(itm['item'])}"):
                        st.write(f"**Target Stream:** {itm['category']}")
                        st.write(f"**Decomposition Period:** {itm['degrade_time']}")
                        st.info(f"💡 **Tip:** {itm['tip']}")

            if last["dets"]:
                st.markdown(section_title("Detected Items Breakdown"), unsafe_allow_html=True)
                cards = "".join(
                    detection_card(i, d["Item"], CATEGORY_LABELS.get(d["Category"], d["Category"]),
                                   CATEGORY_ICONS.get(d["Category"], "❔"), d["Confidence"], d["Certainty"])
                    for i, d in enumerate(last["dets"]))
                st.markdown(f'<div class="det-grid">{cards}</div>', unsafe_allow_html=True)
                with st.expander("Detailed Telemetry Table"):
                    st.dataframe(pd.DataFrame(last["dets"]), use_container_width=True, hide_index=True)
                st.caption("Check the **Smart Bin Finder** tab to navigate to the nearest collection station.")

            st.markdown(section_title("The Three Waste Streams", "The recommended stream for this item is highlighted."),
                        unsafe_allow_html=True)
            render_bins(active=s["category"], counts=Counter(d["Category"] for d in last["dets"]))
        else:
            st.markdown(empty_state_html("Your results will appear here once you analyze a photo or take a camera snapshot."),
                        unsafe_allow_html=True)
            st.markdown(section_title("The Three Waste Streams"), unsafe_allow_html=True)
            render_bins()

# ------------------------------------------------------------------ bin finder
with tab_bins:
    st.markdown(section_title("Campus Smart Bin Finder"), unsafe_allow_html=True)
    st.warning("Bin coordinates are **illustrative demo locations** placed around the campus. "
               "Distances represent direct Haversine calculations.")
    last = st.session_state.get("last")
    default_cat = last["summary"]["category"] if last and last["summary"]["category"] in CATEGORIES else CATEGORIES[0]
    category = st.selectbox("Select Waste Stream", CATEGORIES, index=CATEGORIES.index(default_cat))
    presets = {b.name: (b.lat, b.lon) for b in BINS}
    where = st.selectbox("Your Location Preset", list(presets) + ["Custom coordinates"])
    if where == "Custom coordinates":
        cc1, cc2 = st.columns(2)
        lat = cc1.number_input("Latitude", value=12.9692, format="%.5f")
        lon = cc2.number_input("Longitude", value=79.1559, format="%.5f")
    else:
        lat, lon = presets[where]
    found = nearest_bins(lat, lon, category, top_n=3)
    if found:
        st.success(f"Nearest compatible station for **{category}**: **{found[0][0].name}** "
                   f"({found[0][1]:.0f} m away)")
        st.dataframe(pd.DataFrame([{"Bin Station": b.name, "Distance (m)": round(d), "Accepts Streams": ", ".join(b.accepts)}
                                   for b, d in found]), use_container_width=True, hide_index=True)
    map_rows = [{"lat": b.lat, "lon": b.lon, "color": "#2e7d32" if category in b.accepts else "#9e9e9e"}
                for b in BINS]
    map_rows.append({"lat": lat, "lon": lon, "color": "#d32f2f"})
    st.map(pd.DataFrame(map_rows), latitude="lat", longitude="lon", color="color", size=15)
    st.caption("🟢 Green = Compatible station | ⚪ Grey = Incompatible station | 🔴 Red = Your current location.")

# ------------------------------------------------------------------ history
with tab_hist:
    st.markdown(section_title("Field Journal & Analytics", "Historical telemetry of all analyzed campus items."),
                unsafe_allow_html=True)
    if not HISTORY_PATH.exists():
        st.info("No analyses recorded yet. Scan an item in the Detect tab to populate.")
    else:
        df = pd.read_csv(HISTORY_PATH)
        det = df[df["item"].notna() & (df["item"] != "")].copy()
        det["confidence"] = pd.to_numeric(det["confidence"], errors="coerce")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans", df["analysis_id"].nunique())
        m2.metric("Items Detected", len(det))
        m3.metric("Uncertain Ratio", f"{(det['certainty'] == 'Uncertain').mean():.0%}" if len(det) else "-")
        if len(det):
            a, b = st.columns(2)
            a.subheader("Detections by Category")
            a.bar_chart(det["category"].value_counts().reindex(CATEGORIES, fill_value=0)
                        .rename(index=CATEGORY_LABELS), color="#52b788")
            b.subheader("Top Waste Classes")
            b.bar_chart(det["item"].value_counts().head(10), color="#d9a93a")
            st.subheader("Confidence Operating Statistics")
            st.dataframe(det.groupby("item")["confidence"].agg(["count", "mean", "min", "max"]).round(2)
                         .rename(columns={"count": "Detections", "mean": "Avg Conf", "min": "Min Conf", "max": "Max Conf"}))
        st.subheader("Recent Activity Log")
        st.dataframe(df.tail(50).iloc[::-1], use_container_width=True, hide_index=True)
        if st.checkbox("Enable database reset") and st.button("Clear Log Data"):
            HISTORY_PATH.unlink()
            st.rerun()

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(section_title("System Architecture & Specifications"), unsafe_allow_html=True)
    st.markdown(how_it_works_html(), unsafe_allow_html=True)
    st.subheader("Model Status")
    if METRICS_PATH.exists():
        mt = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        o = mt["overall"]
        st.write(f"Validation performance on **{mt['split']}** split ({mt['num_images']} images):")
        st.table(pd.DataFrame([o]).round(3))
    else:
        st.info("Validation benchmark results will appear after running `python evaluate.py`.")
    st.subheader("Class-to-Stream Rule Table")
    st.dataframe(pd.DataFrame(CLASS_TO_CATEGORY.items(), columns=["Dataset Class", "Target Waste Stream"]),
                 use_container_width=True, hide_index=True)
    st.subheader("Attribution & Licensing")
    st.markdown("Dataset: Waste Detection v3, Roboflow Universe (CC BY 4.0). Built with Ultralytics YOLOv8 & Streamlit.")

st.markdown(footer_html(), unsafe_allow_html=True)
