"""EcoSight AI - Waste categorization, routing logic, and campus bin finder."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

# Canonical Waste Category Constants
ORGANIC = "Organic/Compost"
RECYCLABLE = "Recyclable"
GENERAL = "General Waste"

CATEGORIES = [ORGANIC, RECYCLABLE, GENERAL]

CATEGORY_ICONS = {
    ORGANIC: "🍏",
    RECYCLABLE: "♻️",
    GENERAL: "🗑️",
}

# Detection Confidence Operating Thresholds
CONF_MIN = 0.25
CONF_CONFIDENT = 0.60

# Primary 11 Roboflow Dataset Class Mappings
CLASS_TO_CATEGORY: dict[str, str] = {
    "Compost-Cup": ORGANIC,
    "Compost-Lid": ORGANIC,
    "Compost-Napkin": ORGANIC,
    "Compost-Plate-Bowl": ORGANIC,
    "Compost-Utencil": ORGANIC,
    "Recycle-Aluminum-Can": RECYCLABLE,
    "Recycle-Aluminum-Foil": RECYCLABLE,
    "Recycle-Food-Packaging": RECYCLABLE,
    "Recycle-Plastic-Bottle": RECYCLABLE,
    "Trash-Bowl-Cup": GENERAL,
    "Trash-Wrapper": GENERAL,
}

# COCO Pretrained Dataset Mappings (Graceful fallback when testing with base yolov8n.pt)
COCO_FALLBACK_MAPPING: dict[str, str] = {
    "bottle": RECYCLABLE,
    "can": RECYCLABLE,
    "wine glass": RECYCLABLE,
    "cup": RECYCLABLE,
    "fork": RECYCLABLE,
    "knife": RECYCLABLE,
    "spoon": RECYCLABLE,
    "bowl": GENERAL,
    "banana": ORGANIC,
    "apple": ORGANIC,
    "sandwich": ORGANIC,
    "orange": ORGANIC,
    "broccoli": ORGANIC,
    "carrot": ORGANIC,
    "hot dog": ORGANIC,
    "pizza": ORGANIC,
    "donut": ORGANIC,
    "cake": ORGANIC,
    "book": RECYCLABLE,
    "cell phone": GENERAL,
    "backpack": GENERAL,
    "handbag": GENERAL,
    "cardboard": RECYCLABLE,
}

# Eco-Impact & Degradation Metadata
ECO_METRICS: dict[str, dict[str, str | float]] = {
    "Recycle-Plastic-Bottle": {"co2_saved_g": 85.0, "degrade_time": "450 years", "tip": "Empty remaining liquid and crush bottle."},
    "Recycle-Aluminum-Can": {"co2_saved_g": 130.0, "degrade_time": "200–500 years", "tip": "Can is 100% infinitely recyclable. Ensure no liquid remains."},
    "Recycle-Aluminum-Foil": {"co2_saved_g": 45.0, "degrade_time": "400 years", "tip": "Clean off solid food residue before recycling."},
    "Recycle-Food-Packaging": {"co2_saved_g": 35.0, "degrade_time": "100–300 years", "tip": "Check for grease. If heavily soiled, dispose in General Waste."},
    "Compost-Cup": {"co2_saved_g": 20.0, "degrade_time": "3–6 months", "tip": "Certified compostable. Decomposes naturally in compost stream."},
    "Compost-Lid": {"co2_saved_g": 15.0, "degrade_time": "3–6 months", "tip": "Compostable bioplastic lid."},
    "Compost-Napkin": {"co2_saved_g": 10.0, "degrade_time": "2–4 weeks", "tip": "Can be composted even with minor grease residue."},
    "Compost-Plate-Bowl": {"co2_saved_g": 25.0, "degrade_time": "2–6 months", "tip": "Plant-fiber compostable dishware."},
    "Compost-Utencil": {"co2_saved_g": 15.0, "degrade_time": "6 months", "tip": "CPLA bioplastic utensil."},
    "Trash-Bowl-Cup": {"co2_saved_g": 0.0, "degrade_time": "500+ years", "tip": "Non-recyclable composite or heavily contaminated wax-lined paper."},
    "Trash-Wrapper": {"co2_saved_g": 0.0, "degrade_time": "400+ years", "tip": "Multi-layer foil/plastic laminate wrapper. Landfill/incineration."},
}


@dataclass(frozen=True)
class Bin:
    name: str
    lat: float
    lon: float
    accepts: list[str]


# 10 Illustrative Demo Campus Bins (Centred around VIT Vellore)
BINS: list[Bin] = [
    Bin("Main Gate Eco-Station", 12.96920, 79.15590, [ORGANIC, RECYCLABLE, GENERAL]),
    Bin("Garden Compost Point", 12.97051, 79.15682, [ORGANIC]),
    Bin("Food Court Bin", 12.97284, 79.15820, [ORGANIC, RECYCLABLE, GENERAL]),
    Bin("Library Recycling Station", 12.97120, 79.15940, [RECYCLABLE, GENERAL]),
    Bin("Technology Tower Quad", 12.97010, 79.16010, [ORGANIC, RECYCLABLE, GENERAL]),
    Bin("SJT Cafeteria Hub", 12.97350, 79.16380, [ORGANIC, RECYCLABLE, GENERAL]),
    Bin("Men's Hostel Block D", 12.97510, 79.15720, [ORGANIC, RECYCLABLE, GENERAL]),
    Bin("Ladies Hostel Block G", 12.96780, 79.15810, [ORGANIC, RECYCLABLE, GENERAL]),
    Bin("Sports Complex Point", 12.96640, 79.15420, [RECYCLABLE, GENERAL]),
    Bin("Main Canteen Segregator", 12.96850, 79.15660, [ORGANIC, RECYCLABLE, GENERAL]),
]


def category_for(item_name: str) -> str:
    """Map class label to canonical waste category, with fallback prefixes and COCO support."""
    if item_name in CLASS_TO_CATEGORY:
        return CLASS_TO_CATEGORY[item_name]

    low = item_name.strip().lower()
    if low in COCO_FALLBACK_MAPPING:
        return COCO_FALLBACK_MAPPING[low]

    # Prefix heuristic
    if low.startswith("compost") or "organic" in low or "food" in low:
        return ORGANIC
    if low.startswith("recycle") or "plastic" in low or "can" in low or "bottle" in low or "aluminum" in low:
        return RECYCLABLE
    if low.startswith("trash") or "waste" in low or "wrapper" in low:
        return GENERAL

    return GENERAL


def confidence_label(conf: float) -> str:
    """Label prediction as Confident if >= 0.60, else Uncertain."""
    return "Confident" if conf >= CONF_CONFIDENT else "Uncertain"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate straight-line distance (in meters) between two coordinates using the Haversine formula."""
    r = 6371000.0  # Earth's mean radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def nearest_bins(lat: float, lon: float, category: str, top_n: int = 3) -> list[tuple[Bin, float]]:
    """Return top_n nearest bins accepting the specified waste category."""
    candidates = [b for b in BINS if category in b.accepts]
    ranked = [(b, haversine_distance(lat, lon, b.lat, b.lon)) for b in candidates]
    ranked.sort(key=lambda pair: pair[1])
    return ranked[:top_n]


def summarize(dets: list[dict]) -> dict:
    """Summarize detections into a winning category verdict and status message."""
    if not dets:
        return {
            "status": "none",
            "category": None,
            "message": "No waste items detected. Try moving closer, improving lighting, or lowering the confidence threshold.",
        }

    # Aggregate confidences by category: S(c) = sum(conf(d))
    category_scores: dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    max_conf_by_category: dict[str, float] = {cat: 0.0 for cat in CATEGORIES}

    for d in dets:
        cat = d.get("Category", GENERAL)
        if cat not in category_scores:
            cat = GENERAL
        c = float(d.get("Confidence", 0.0))
        category_scores[cat] += c
        if c > max_conf_by_category[cat]:
            max_conf_by_category[cat] = c

    # Find dominant category
    winning_category = max(category_scores.keys(), key=lambda k: category_scores[k])
    total_distinct_cats = sum(1 for c, score in category_scores.items() if score > 0)
    top_conf = max_conf_by_category[winning_category]

    is_confident = top_conf >= CONF_CONFIDENT
    status = "confident" if is_confident else "uncertain"

    multi_item_note = " Multiple waste categories detected — please sort and dispose of each item separately." if total_distinct_cats > 1 else ""

    if is_confident:
        msg = f"Primary Category: {winning_category} (Top confidence: {top_conf:.0%}).{multi_item_note}"
    else:
        msg = f"Potential Category: {winning_category} (Confidence: {top_conf:.0%}). Uncertain detection — verify item before disposal.{multi_item_note}"

    return {
        "status": status,
        "category": winning_category,
        "message": msg,
    }


def get_impact_summary(dets: list[dict]) -> dict:
    """Compute carbon offset and decomposition facts for detected items."""
    total_co2 = 0.0
    items_breakdown = []
    for d in dets:
        item = d.get("Item", "")
        cat = d.get("Category", GENERAL)
        info = ECO_METRICS.get(item, {
            "co2_saved_g": 50.0 if cat == RECYCLABLE else (15.0 if cat == ORGANIC else 0.0),
            "degrade_time": "100+ years" if cat != ORGANIC else "2–6 months",
            "tip": "Sort into designated container."
        })
        co2 = float(info["co2_saved_g"])
        total_co2 += co2
        items_breakdown.append({
            "item": item,
            "category": cat,
            "co2_saved_g": co2,
            "degrade_time": info["degrade_time"],
            "tip": info["tip"],
        })
    return {
        "total_co2_saved_g": round(total_co2, 1),
        "items": items_breakdown,
    }
