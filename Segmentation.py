import streamlit as st
import pandas as pd
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Topbar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2rem;
    background: rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
    border-radius: 0 0 20px 20px;
}
.topbar-logo {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.topbar-sub {
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
}

/* ── Panel titles ── */
.panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Glass panel ── */
.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(14px);
    height: 100%;
}

/* ── Input labels ── */
label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
input[type="number"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
input[type="number"]:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.18) !important;
}

/* ── Group header inside form ── */
.input-group-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1rem 0 0.4rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 24px rgba(109,40,217,0.45) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(109,40,217,0.65) !important;
}

/* ── Placeholder card (before predict) ── */
.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 160px;
    border: 2px dashed rgba(167,139,250,0.25);
    border-radius: 16px;
    color: #475569;
    font-size: 0.9rem;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}
.placeholder .ph-icon { font-size: 2.2rem; }

/* ── Result banner ── */
.result-banner {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    animation: slideUp 0.4s ease;
}
.result-banner .cluster-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 6px;
    padding: 0.22rem 0.7rem;
    margin-bottom: 0.7rem;
}
.result-banner h2 {
    font-size: 1.5rem;
    font-weight: 800;
    margin: 0 0 0.35rem;
}
.result-banner p {
    font-size: 0.88rem;
    margin: 0;
    opacity: 0.85;
    line-height: 1.6;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Segment grid ── */
.seg-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
}
.seg-item {
    border-radius: 12px;
    padding: 1rem 1.1rem;
    border: 1px solid;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.seg-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.seg-item .si-badge {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 5px;
    padding: 0.18rem 0.55rem;
    display: inline-block;
    margin-bottom: 0.45rem;
}
.seg-item h5 {
    font-size: 0.88rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.2rem;
}
.seg-item p {
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
}

/* ── Stat chips ── */
.stat-row {
    display: flex;
    gap: 0.7rem;
    margin-top: 1.4rem;
}
.stat-chip {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.7rem 0.5rem;
    text-align: center;
}
.stat-chip .sv {
    font-size: 1.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-chip .sl {
    font-size: 0.68rem;
    color: #475569;
    font-weight: 600;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

hr.div { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ── Segment definitions ────────────────────────────────────────────────────────
SEGMENTS = {
    0: {
        "name": "😴 Low Engagement",
        "desc": "Older, low-income, very low spenders with high web visits but rarely buy. High recency — long since last purchase.",
        "bg":     "linear-gradient(135deg,rgba(148,163,184,.18),rgba(148,163,184,.06))",
        "border": "#334155",
        "badge_bg": "#1e293b", "badge_fg": "#94a3b8",
    },
    1: {
        "name": "💎 Premium In-Store",
        "desc": "High income, very high spending, mostly in-store buyers. Loyal and frequent with moderate recency.",
        "bg":     "linear-gradient(135deg,rgba(234,179,8,.18),rgba(234,179,8,.06))",
        "border": "#92400e",
        "badge_bg": "#78350f", "badge_fg": "#fde68a",
    },
    2: {
        "name": "🌐 Active Digital",
        "desc": "Mid-income, moderate spenders who prefer online channels. High web visits and very low recency — recently active.",
        "bg":     "linear-gradient(135deg,rgba(14,165,233,.18),rgba(14,165,233,.06))",
        "border": "#0c4a6e",
        "badge_bg": "#0c4a6e", "badge_fg": "#7dd3fc",
    },
    3: {
        "name": "👑 High-Value Senior",
        "desc": "Oldest group, highest income, high spending, prefer in-store. Low web visits — offline-focused premium buyers.",
        "bg":     "linear-gradient(135deg,rgba(168,85,247,.18),rgba(168,85,247,.06))",
        "border": "#6b21a8",
        "badge_bg": "#581c87", "badge_fg": "#e9d5ff",
    },
    4: {
        "name": "🛒 Omnichannel Buyer",
        "desc": "Mid-income, balanced spending across both web and store. Highest web purchase rate — comfortable in all channels.",
        "bg":     "linear-gradient(135deg,rgba(20,184,166,.18),rgba(20,184,166,.06))",
        "border": "#115e59",
        "badge_bg": "#134e4a", "badge_fg": "#99f6e4",
    },
    5: {
        "name": "💤 Dormant / Budget",
        "desc": "Lowest income and spending of all clusters. Very low purchase activity across all channels. Hardest to re-engage.",
        "bg":     "linear-gradient(135deg,rgba(236,72,153,.18),rgba(236,72,153,.06))",
        "border": "#831843",
        "badge_bg": "#831843", "badge_fg": "#f9a8d4",
    },
}

def seg_info(cid):
    return SEGMENTS.get(cid, {
        "name": f"Cluster {cid}", "desc": "A unique customer segment.",
        "bg": "linear-gradient(135deg,rgba(148,163,184,.15),rgba(148,163,184,.05))",
        "border": "#334155", "badge_bg": "#1e293b", "badge_fg": "#94a3b8",
    })

# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">🎯 SegmentIQ</div>
    <div class="topbar-sub">K-Means · 7 Features · Real-time Prediction</div>
</div>
""", unsafe_allow_html=True)

# ── Two-column dashboard layout ───────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

# ────────────────────────────── LEFT: Input Panel ─────────────────────────────
with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📋 Customer Profile</div>', unsafe_allow_html=True)

    # — Demographics —
    st.markdown('<div class="input-group-header">Demographics</div>', unsafe_allow_html=True)
    age    = st.number_input("Age", min_value=18, max_value=100, value=35, key="age")
    income = st.number_input("Annual Income ($)", min_value=0, max_value=200000,
                              value=50000, step=1000, key="income")

    # — Purchase Behaviour —
    st.markdown('<div class="input-group-header">Purchase Behaviour</div>', unsafe_allow_html=True)
    total_spending     = st.number_input("Total Spending ($)", min_value=0, max_value=5000,
                                          value=1000, step=50, key="spend")
    num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100,
                                           value=10, key="store")
    recency            = st.number_input("Recency (days since last purchase)",
                                          min_value=0, max_value=365, value=30, key="recency")

    # — Online Activity —
    st.markdown('<div class="input-group-header">Online Activity</div>', unsafe_allow_html=True)
    num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100,
                                         value=10, key="web_p")
    num_web_visits    = st.number_input("Web Visits / Month", min_value=0, max_value=50,
                                         value=3, key="web_v")

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    predict = st.button("🔍 Predict Segment", use_container_width=True)

    # Stat chips
    st.markdown("""
    <div class="stat-row">
        <div class="stat-chip"><div class="sv">6</div><div class="sl">Segments</div></div>
        <div class="stat-chip"><div class="sv">7</div><div class="sl">Features</div></div>
        <div class="stat-chip"><div class="sv">2216</div><div class="sl">Customers</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────── RIGHT: Results + Guide ───────────────────────────
with right:

    # ── Result section ──
    st.markdown('<div class="panel-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    if predict:
        input_data = pd.DataFrame({
            "Age": [age], "Income": [income],
            "Total_Spending": [total_spending],
            "NumWebPurchases": [num_web_purchases],
            "NumStorePurchases": [num_store_purchases],
            "NumWebVisitsMonth": [num_web_visits],
            "Recency": [recency],
        })
        cluster = kmeans.predict(scaler.transform(input_data))[0]
        seg = seg_info(cluster)

        st.markdown(f"""
        <div class="result-banner" style="background:{seg['bg']};
             border:1px solid {seg['border']};">
            <span class="cluster-badge"
                  style="background:{seg['badge_bg']}; color:{seg['badge_fg']};">
                Cluster {cluster}
            </span>
            <h2 style="color:#f1f5f9;">{seg['name']}</h2>
            <p style="color:#cbd5e1;">{seg['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder">
            <span class="ph-icon">🎯</span>
            <span>Fill in the profile & click <strong>Predict Segment</strong></span>
        </div>
        """, unsafe_allow_html=True)

    # ── Segment reference grid ──
    st.markdown('<div class="panel-title" style="margin-top:0.5rem;">🗺️ All Segments</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="seg-grid">', unsafe_allow_html=True)
    for cid, s in SEGMENTS.items():
        st.markdown(f"""
        <div class="seg-item" style="background:{s['bg']}; border-color:{s['border']};">
            <span class="si-badge"
                  style="background:{s['badge_bg']}; color:{s['badge_fg']};">
                Cluster {cid}
            </span>
            <h5>{s['name']}</h5>
            <p>{s['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#334155;font-size:0.78rem;">
    SegmentIQ · Built with Streamlit · K-Means Clustering
</div>
""", unsafe_allow_html=True)
