"""EcoSight AI - Custom UI Theme, Glassmorphic Styling, and HTML Component Renderers."""
from __future__ import annotations

import html


def friendly_name(cls_name: str) -> str:
    """Format technical dataset class strings into user-friendly names."""
    if not cls_name:
        return "Unknown Item"
    cleaned = cls_name.replace("Compost-", "").replace("Recycle-", "").replace("Trash-", "")
    return cleaned.replace("-", " ").title()


def theme_css(motion: bool = True) -> str:
    anim_style = """
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    """ if motion else ""

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Global Hero & Badges */
    .hero-banner {{
        background: linear-gradient(135deg, rgba(27, 67, 50, 0.95), rgba(45, 106, 79, 0.90)),
                    radial-gradient(circle at top right, rgba(82, 183, 136, 0.4), transparent);
        color: #f8fafc;
        padding: 32px 36px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(27, 67, 50, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .hero-title {{
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: #d8f3dc;
        margin: 0 0 16px 0;
        max-width: 720px;
        line-height: 1.5;
    }}
    .hero-tags {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }}
    .hero-tag {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 5px 14px;
        border-radius: 100px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    /* Verdict Alert Box */
    .verdict {{
        padding: 18px 24px;
        border-radius: 14px;
        margin: 18px 0 24px 0;
        font-size: 1.05rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        {anim_style}
    }}
    .v-confident {{
        background: #eafaf1;
        border-left: 6px solid #2d6a4f;
        color: #1b4332;
    }}
    .v-uncertain {{
        background: #fff8e6;
        border-left: 6px solid #d97706;
        color: #78350f;
    }}
    .v-none {{
        background: #f1f5f9;
        border-left: 6px solid #64748b;
        color: #334155;
    }}

    /* Section Titles */
    .sec-wrap {{
        margin: 28px 0 16px 0;
    }}
    .sec-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }}
    .sec-subtitle {{
        font-size: 0.92rem;
        color: #64748b;
        margin: 0;
    }}

    /* Detection Cards Grid */
    .det-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px;
        margin-top: 14px;
        margin-bottom: 20px;
    }}
    .det-card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        {anim_style}
    }}
    .det-card:hover {{
        border-color: #52b788;
        box-shadow: 0 8px 24px rgba(45, 106, 79, 0.12);
        transform: translateY(-2px);
    }}
    .det-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}
    .det-name {{
        font-weight: 700;
        font-size: 1.05rem;
        color: #0f172a;
    }}
    .det-category-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        margin-bottom: 12px;
    }}
    .det-meter-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 6px;
    }}
    .det-meter-bg {{
        height: 8px;
        background: #f1f5f9;
        border-radius: 10px;
        overflow: hidden;
    }}
    .det-meter-bar {{
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #40916c, #52b788);
    }}
    .chip-confident {{
        background: #d8f3dc;
        color: #1b4332;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }}
    .chip-uncertain {{
        background: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }}

    /* Bin Cards Grid */
    .bin-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        margin: 16px 0 28px 0;
    }}
    @media (max-width: 860px) {{
        .bin-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    .bin-box {{
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        position: relative;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        {anim_style}
    }}
    .bin-box.active {{
        border-color: #2d6a4f;
        background: linear-gradient(180deg, #f7fcf9, #ffffff);
        box-shadow: 0 8px 30px rgba(45, 106, 79, 0.18);
        transform: scale(1.02);
    }}
    .bin-badge-active {{
        position: absolute;
        top: -12px;
        right: 18px;
        background: #2d6a4f;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 100px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 4px 10px rgba(45, 106, 79, 0.3);
    }}
    .bin-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }}
    .bin-icon {{
        font-size: 1.8rem;
    }}
    .bin-title {{
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }}
    .bin-count {{
        font-size: 0.88rem;
        color: #2d6a4f;
        font-weight: 700;
        margin-bottom: 10px;
    }}
    .bin-items-title {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 700;
        margin: 8px 0 6px 0;
    }}
    .bin-pill-list {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }}
    .bin-pill {{
        background: #f1f5f9;
        color: #334155;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 6px;
    }}

    /* Empty States & Footers */
    .empty-state {{
        padding: 42px 24px;
        text-align: center;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        background: #f8fafc;
        margin: 18px 0;
    }}
    .empty-state-icon {{
        font-size: 2.4rem;
        margin-bottom: 8px;
    }}
    .empty-state-text {{
        color: #64748b;
        font-weight: 500;
        font-size: 1.0rem;
    }}

    .footer-box {{
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
    }}
    </style>
    """


def background_html() -> str:
    """Subtle SVG aesthetic background decor."""
    return ""


def hero_html() -> str:
    """Render hero header banner."""
    return """
    <div class="hero-banner">
        <div class="hero-title">
            <span>🌿</span> EcoSight AI
        </div>
        <div class="hero-subtitle">
            Autonomous Computer-Vision Smart Waste Segregation & Campus Disposal Guidance.
            Real-time inference mapped across Organic, Recyclable, and General streams.
        </div>
        <div class="hero-tags">
            <span class="hero-tag">⚡ YOLOv8n Transfer Learning</span>
            <span class="hero-tag">📍 Smart Bin Haversine Routing</span>
            <span class="hero-tag">🎯 Uncertainty Calibration (&lt;60%)</span>
            <span class="hero-tag">🌱 Carbon Offset Tracking</span>
        </div>
    </div>
    """


def section_title(title: str, subtitle: str = "") -> str:
    sub_tag = f'<p class="sec-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    return f"""
    <div class="sec-wrap">
        <h3 class="sec-title">{html.escape(title)}</h3>
        {sub_tag}
    </div>
    """


def detection_card(i: int, item: str, category_label: str, icon: str, conf: float, certainty: str) -> str:
    chip_class = "chip-confident" if certainty == "Confident" else "chip-uncertain"
    conf_pct = int(round(conf * 100))
    bar_width = max(5, min(100, conf_pct))
    clean_item = friendly_name(item)

    return f"""
    <div class="det-card">
        <div class="det-top">
            <span class="det-name">{html.escape(clean_item)}</span>
            <span class="{chip_class}">{html.escape(certainty)}</span>
        </div>
        <div class="det-category-pill">
            <span>{icon}</span> {html.escape(category_label)}
        </div>
        <div class="det-meter-label">
            <span>Detection Confidence</span>
            <span>{conf:.1%}</span>
        </div>
        <div class="det-meter-bg">
            <div class="det-meter-bar" style="width: {bar_width}%;"></div>
        </div>
    </div>
    """


def bin_cards(cards: list[dict]) -> str:
    rendered = []
    for c in cards:
        active_class = " active" if c.get("active") else ""
        badge = '<div class="bin-badge-active">⭐ Recommended</div>' if c.get("active") else ""
        count = c.get("count", 0)
        count_label = f"• {count} item{'s' if count != 1 else ''} detected" if count > 0 else "• Ready for disposal"
        items_html = "".join(f'<span class="bin-pill">{html.escape(it)}</span>' for it in c.get("items", []))

        rendered.append(f"""
        <div class="bin-box{active_class}">
            {badge}
            <div class="bin-header">
                <span class="bin-icon">{c.get('icon', '📦')}</span>
                <h4 class="bin-title">{html.escape(c.get('title', 'Bin'))}</h4>
            </div>
            <div class="bin-count">{count_label}</div>
            <div class="bin-items-title">Accepted Items:</div>
            <div class="bin-pill-list">{items_html}</div>
        </div>
        """)
    return f'<div class="bin-grid">{"".join(rendered)}</div>'


def empty_state_html(message: str) -> str:
    return f"""
    <div class="empty-state">
        <div class="empty-state-icon">📷</div>
        <div class="empty-state-text">{html.escape(message)}</div>
    </div>
    """


def how_it_works_html() -> str:
    return """
    <div style="background: #ffffff; border-radius: 16px; padding: 24px; border: 1px solid #e2e8f0; line-height: 1.7; color: #334155;">
        <h4 style="margin-top: 0; color: #1e293b;">1. YOLOv8 Deep Object Detection</h4>
        <p>The system evaluates incoming image tensors at 640×640 px resolution. Feature pyramids extract high-level semantic contours to pinpoint waste items even in cluttered campus backgrounds.</p>

        <h4 style="color: #1e293b;">2. Three-Stream Semantic Classification</h4>
        <p>Fine-grained classes are mapped into <strong>Organic/Compost</strong>, <strong>Recyclable</strong>, and <strong>General Waste</strong> using strict contamination boundary rules.</p>

        <h4 style="color: #1e293b;">3. Uncertainty & False-Positive Mitigation</h4>
        <p>Detections below <strong>60% confidence</strong> are marked <em>Uncertain</em>, preventing improper stream contamination.</p>

        <h4 style="color: #1e293b;">4. Geo-Proximity Smart Routing</h4>
        <p>Using the Haversine great-circle formula, EcoSight calculates the shortest walking distance to campus disposal hubs.</p>
    </div>
    """


def footer_html() -> str:
    return """
    <div class="footer-box">
        EcoSight AI — Campus Smart Waste Segregation System &bull; Powered by YOLOv8 &bull; CC BY 4.0 Attribution
    </div>
    """
