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
                             nearest_bins, summarize)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
HISTORY_PATH = ROOT / "results" / "history.csv"
METRICS_PATH = ROOT / "results" / "metrics.json"

# Display-only labels; the category values used by the logic are unchanged.
CATEGORY_LABELS = {ORGANIC: "Organic", RECYCLABLE: "Recyclable", GENERAL: "General / Non-recyclable"}
BIN_CSS = {ORGANIC: "bin-organic", RECYCLABLE: "bin-recycle", GENERAL: "bin-general"}

st.set_page_config(page_title="EcoSight AI", page_icon="🌿", layout="wide")


def keyed_container(key: str):
    """Container with a stable CSS class (st-key-<key>); falls back to a plain container on old Streamlit."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


def run_detection(model, image: Image.Image, conf: float, iou: float):
    bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])
    result = model.predict(bgr, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    annotated = np.ascontiguousarray(result.plot()[:, :, ::-1])
    dets = []
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        c = float(box.conf[0])
        dets.append({"Item": name, "Category": category_for(name) or "Unmapped",
                     "Confidence": round(c, 3), "Certainty": confidence_label(c)})
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
st.sidebar.title("🌿 EcoSight")
st.sidebar.caption("Computer-vision waste segregation for campus waste management")
conf_thr = st.sidebar.slider("Minimum confidence", 0.10, 0.90, CONF_MIN, 0.05)
iou_thr = st.sidebar.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
model_ready = MODEL_PATH.exists()
if model_ready:
    st.sidebar.success("Trained weights found: models/best.pt")
else:
    st.sidebar.error("models/best.pt not found - train the model first (see README).")
st.sidebar.info(f"Detections below {CONF_CONFIDENT:.0%} confidence are flagged **Uncertain**.")
motion = st.sidebar.toggle("Gentle animations", value=True)

# ------------------------------------------------------------------ hero
st.markdown(theme_css(motion) + background_html() + hero_html(), unsafe_allow_html=True)

tab_detect, tab_bins, tab_hist, tab_about = st.tabs(
    ["🌿 Detect", "🗺️ Bin Finder", "📖 Field Journal", "🍃 About"])

# ------------------------------------------------------------------ detect
with tab_detect:
    if not model_ready:
        st.error("No trained model available, so detection is disabled. "
                 "Run `python train.py --mode full` then reload this page.")
    else:
        st.markdown(section_title("Show us what you found",
                                  "Upload a photo and the detector will look for the 11 waste classes it was trained on."),
                    unsafe_allow_html=True)
        with keyed_container("upload_card"):
            up = st.file_uploader("Upload a photo of waste items", type=["jpg", "jpeg", "png"])
            if up is not None:
                image = Image.open(up)
                if st.button("Analyze image", type="primary"):
                    annotated, dets = run_detection(load_model(), image, conf_thr, iou_thr)
                    save_history(up.name, dets)
                    st.session_state["last"] = {"annotated": annotated, "original": image,
                                                "dets": dets, "summary": summarize(dets)}
        last = st.session_state.get("last")
        if last:
            c1, c2 = st.columns(2)
            c1.image(last["original"], caption="Original", use_container_width=True)
            c2.image(last["annotated"], caption="Detections", use_container_width=True)
            s = last["summary"]
            icon = CATEGORY_ICONS.get(s["category"], "❔")
            st.markdown(f'<div class="verdict v-{s["status"]}">{icon} {html.escape(s["message"])}</div>',
                        unsafe_allow_html=True)
            if last["dets"]:
                st.markdown(section_title("What the forest found"), unsafe_allow_html=True)
                cards = "".join(
                    detection_card(i, d["Item"], CATEGORY_LABELS.get(d["Category"], d["Category"]),
                                   CATEGORY_ICONS.get(d["Category"], "❔"), d["Confidence"], d["Certainty"])
                    for i, d in enumerate(last["dets"]))
                st.markdown(f'<div class="det-grid">{cards}</div>', unsafe_allow_html=True)
                with st.expander("Detection table"):
                    st.dataframe(pd.DataFrame(last["dets"]), use_container_width=True, hide_index=True)
                st.caption("Go to the **Bin Finder** tab to locate the nearest compatible bin.")
            st.markdown(section_title("The three bins", "The suggested bin is highlighted."),
                        unsafe_allow_html=True)
            render_bins(active=s["category"], counts=Counter(d["Category"] for d in last["dets"]))
        else:
            st.markdown(empty_state_html("Your results will appear here once you analyze a photo."),
                        unsafe_allow_html=True)
            st.markdown(section_title("The three bins"), unsafe_allow_html=True)
            render_bins()

# ------------------------------------------------------------------ bin finder
with tab_bins:
    st.markdown(section_title("Find the nearest bin"), unsafe_allow_html=True)
    st.warning("Bin locations are **illustrative demo coordinates**, not verified campus infrastructure. "
               "Distances are straight-line estimates.")
    last = st.session_state.get("last")
    default_cat = last["summary"]["category"] if last and last["summary"]["category"] in CATEGORIES else CATEGORIES[0]
    category = st.selectbox("Waste category", CATEGORIES, index=CATEGORIES.index(default_cat))
    presets = {b.name: (b.lat, b.lon) for b in BINS}
    where = st.selectbox("Your location (demo)", list(presets) + ["Custom coordinates"])
    if where == "Custom coordinates":
        cc1, cc2 = st.columns(2)
        lat = cc1.number_input("Latitude", value=12.9692, format="%.5f")
        lon = cc2.number_input("Longitude", value=79.1559, format="%.5f")
    else:
        lat, lon = presets[where]
    found = nearest_bins(lat, lon, category, top_n=3)
    if found:
        st.success(f"Nearest compatible bin for **{category}**: **{found[0][0].name}** "
                   f"({found[0][1]:.0f} m away)")
        st.dataframe(pd.DataFrame([{"Bin": b.name, "Distance (m)": round(d), "Accepts": ", ".join(b.accepts)}
                                   for b, d in found]), use_container_width=True, hide_index=True)
    map_rows = [{"lat": b.lat, "lon": b.lon, "color": "#2e7d32" if category in b.accepts else "#9e9e9e"}
                for b in BINS]
    map_rows.append({"lat": lat, "lon": lon, "color": "#d32f2f"})
    st.map(pd.DataFrame(map_rows), latitude="lat", longitude="lon", color="color", size=15)
    st.caption("Green = compatible bin, grey = incompatible, red = your location.")

# ------------------------------------------------------------------ history
with tab_hist:
    st.markdown(section_title("Field Journal", "Every analysis you run is recorded here."),
                unsafe_allow_html=True)
    if not HISTORY_PATH.exists():
        st.info("No analyses yet. Upload an image in the Detect tab.")
    else:
        df = pd.read_csv(HISTORY_PATH)
        det = df[df["item"].notna() & (df["item"] != "")].copy()
        det["confidence"] = pd.to_numeric(det["confidence"], errors="coerce")
        m1, m2, m3 = st.columns(3)
        m1.metric("Images analyzed", df["analysis_id"].nunique())
        m2.metric("Items detected", len(det))
        m3.metric("Uncertain share", f"{(det['certainty'] == 'Uncertain').mean():.0%}" if len(det) else "-")
        if len(det):
            a, b = st.columns(2)
            a.subheader("Detections by category")
            a.bar_chart(det["category"].value_counts().reindex(CATEGORIES, fill_value=0)
                        .rename(index=CATEGORY_LABELS), color="#5f8a4b")
            b.subheader("Detections by item")
            b.bar_chart(det["item"].value_counts(), color="#d9a93a")
            st.subheader("Average confidence per item")
            st.dataframe(det.groupby("item")["confidence"].agg(["count", "mean"]).round(2)
                         .rename(columns={"count": "Detections", "mean": "Avg confidence"}))
        st.subheader("Recent records")
        st.dataframe(df.tail(50).iloc[::-1], use_container_width=True, hide_index=True)
        if st.checkbox("I want to clear all history") and st.button("Clear history"):
            HISTORY_PATH.unlink()
            st.rerun()

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(section_title("How EcoSight works"), unsafe_allow_html=True)
    st.markdown(how_it_works_html(), unsafe_allow_html=True)
    st.subheader("Model status")
    if METRICS_PATH.exists():
        mt = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        o = mt["overall"]
        st.write(f"Measured on the **{mt['split']}** split ({mt['num_images']} images), "
                 f"evaluated {mt['evaluated_at']}:")
        st.table(pd.DataFrame([o]).round(3))
        st.caption("The validation set is small, so these numbers are noisy.")
    else:
        st.info("No evaluation results yet. Run `python evaluate.py` to generate real metrics.")
    st.subheader("Class → category mapping")
    st.dataframe(pd.DataFrame(CLASS_TO_CATEGORY.items(), columns=["Dataset class", "Category"]),
                 use_container_width=True, hide_index=True)
    st.subheader("Limitations")
    st.markdown("- Only the 11 dataset classes can be recognised; other objects may be missed or mislabelled.\n"
                "- Mapping to categories is a fixed rule and may differ from local campus rules.\n"
                "- Bin locations are illustrative demo data.\n"
                "- Accuracy depends on the actual training run and a small validation set.")
    st.subheader("Dataset attribution")
    st.markdown("Waste Detection v3, Roboflow Universe "
                "([link](https://universe.roboflow.com/test-mq1p0/waste-detection-v3-tunsu/dataset/7)), "
                "licensed CC BY 4.0.")

st.markdown(footer_html(), unsafe_allow_html=True)