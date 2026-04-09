"""
app.py
------
MAILSAFE AI — Behavioral Anomaly Dashboard
Upload a raw Enron-style CSV (columns: 'file', 'message') and the app
runs batch anomaly profiling: parse → feature engineer → score against pre-trained model → results.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, csv, sys, re, ast
from pathlib import Path
from email.parser import Parser as EmailParser
from email.utils import parsedate_to_datetime
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAILSAFE AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    h1 { color: #e94560 !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "total_emails_sent", "unique_recipients", "avg_recipients_per_email",
    "emails_per_day", "night_email_ratio", "weekend_ratio",
    "burst_score", "recipient_entropy", "avg_subject_length",
    "response_concentration",
]


# ══════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS (mirroring src/ modules but embedded
# so the app is self-contained)
# ══════════════════════════════════════════════════════════

def parse_raw_csv(file_bytes: bytes, max_emails: int = 50_000) -> pd.DataFrame:
    """Parse a raw Enron-style CSV (columns: file, message) directly from bytes."""
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    parser = EmailParser()
    records = []

    reader = csv.DictReader(io.TextIOWrapper(io.BytesIO(file_bytes), encoding="utf-8", errors="replace"))
    for i, row in enumerate(reader):
        if max_emails and i >= max_emails:
            break
        raw = row.get("message", "")
        if not raw:
            continue

        msg        = parser.parsestr(raw)
        sender     = msg.get("From", "").strip().lower()
        to_raw     = msg.get("To", "") or ""
        cc_raw     = msg.get("Cc", "") or ""
        date_raw   = msg.get("Date", "")
        subject    = msg.get("Subject", "")

        to_list    = [r.strip().lower() for r in re.split(r"[,;]", to_raw) if r.strip()]
        cc_list    = [r.strip().lower() for r in re.split(r"[,;]", cc_raw) if r.strip()]
        recipients = list(set(to_list + cc_list))

        timestamp = None
        if date_raw:
            try:
                timestamp = parsedate_to_datetime(date_raw).replace(tzinfo=None)
            except Exception:
                pass

        if not sender or "@" not in sender or timestamp is None:
            continue

        records.append({
            "sender": sender,
            "recipients": recipients,
            "num_recipients": len(recipients),
            "timestamp": timestamp,
            "subject_len": len(subject),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def _recipient_entropy(series) -> float:
    all_r = []
    for v in series:
        if isinstance(v, list):
            all_r.extend(v)
    if not all_r:
        return 0.0
    counts = Counter(all_r)
    total  = sum(counts.values())
    probs  = [c / total for c in counts.values()]
    return float(scipy_entropy(probs))


def build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate parsed emails into per-sender behavioural profiles."""
    df = df.copy()
    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["date"]    = df["timestamp"].dt.date
    df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    profiles = []
    for sender, grp in df.groupby("sender"):
        n = len(grp)
        if n < 2:
            continue
        date_range     = max((grp["timestamp"].max() - grp["timestamp"].min()).days + 1, 1)
        avg_recip      = grp["num_recipients"].mean()
        all_recips     = [r for lst in grp["recipients"] for r in (lst if isinstance(lst, list) else [])]
        unique_recip   = len(set(all_recips))
        recip_entropy  = _recipient_entropy(grp["recipients"])
        top_count      = Counter(all_recips).most_common(1)[0][1] if all_recips else 0
        concentration  = top_count / max(len(all_recips), 1)
        hourly         = grp.groupby(grp["timestamp"].dt.floor("h")).size()
        burst_score    = float(hourly.std()) if len(hourly) > 1 else 0.0

        profiles.append({
            "sender":                    sender,
            "total_emails_sent":         n,
            "unique_recipients":         unique_recip,
            "avg_recipients_per_email":  round(avg_recip, 3),
            "emails_per_day":            round(n / date_range, 4),
            "night_email_ratio":         round(grp["is_night"].mean(), 4),
            "weekend_ratio":             round(grp["is_weekend"].mean(), 4),
            "burst_score":               round(burst_score, 4),
            "recipient_entropy":         round(recip_entropy, 4),
            "avg_subject_length":        round(grp["subject_len"].mean(), 2),
            "response_concentration":    round(concentration, 4),
        })

    return pd.DataFrame(profiles).set_index("sender")


def run_model(profiles: pd.DataFrame, contamination: float = 0.10) -> pd.DataFrame:
    """Score profiles using the pre-trained Isolation Forest."""
    # Load pre-trained models
    model_dir = Path(__file__).parent / "models"
    try:
        scaler = joblib.load(model_dir / "scaler.pkl")
        model = joblib.load(model_dir / "isolation_forest.pkl")
    except FileNotFoundError:
        st.error("Pre-trained model files not found in the 'models/' directory. Cannot score senders.")
        st.stop()

    # Pre-trained model handles scaling, but we need the original features
    X = scaler.transform(profiles[FEATURE_COLS])
    
    raw = model.decision_function(X)
    labels = model.predict(X)                   # -1 = anomaly
    
    # Calculate anomaly score using min/max from the original training run
    # For a robust proxy, we use the theoretical range of IF decision function roughly [-0.5, 0.5] if min/max aren't saved
    # or we can approximate using the current batch min/max for display purposes if the batch is large enough.
    # To keep scores comparable, it's best to use fixed min/max or the batch min/max if necessary. Let's use batch for now if we don't have saved limits.
    mn, mx = raw.min(), raw.max()
    scores = 100 * (1 - (raw - mn) / (mx - mn + 1e-9))

    pca    = PCA(n_components=2, random_state=42)
    # Handle cases where there might be fewer than 2 samples
    if len(X) >= 2:
        coords = pca.fit_transform(X)
    else:
        coords = np.zeros((len(X), 2))

    out                    = profiles.copy()
    out["anomaly_score"]   = np.round(scores, 2)
    out["is_anomaly"]      = (labels == -1).astype(int)
    out["pca_x"]           = coords[:, 0]
    out["pca_y"]           = coords[:, 1]
    return out.sort_values("anomaly_score", ascending=False)


# ══════════════════════════════════════════════════════════
# DEMO DATA (if user hasn't uploaded anything yet)
# ══════════════════════════════════════════════════════════

@st.cache_data
def demo_results() -> pd.DataFrame:
    np.random.seed(42)
    n_n, n_a = 180, 20
    normal = {c: np.random.uniform(*lo_hi, n_n) for c, lo_hi in zip(FEATURE_COLS, [
        (5,80), (3,40), (1,3.5), (0.5,5), (0,0.15), (0.05,0.3),
        (0.1,2), (1.5,4), (20,60), (0.1,0.4)])}
    anomal = {c: np.random.uniform(*lo_hi, n_a) for c, lo_hi in zip(FEATURE_COLS, [
        (300,2000), (200,1500), (15,80), (50,300), (0.4,0.9), (0.5,0.95),
        (10,50), (0.1,1.2), (5,20), (0.6,1.0)])}

    df = pd.DataFrame({k: np.concatenate([normal[k], anomal[k]]) for k in FEATURE_COLS})
    df.insert(0, "sender",
              [f"user_{i:03d}@enron.com" for i in range(n_n)] +
              [f"system_{i:02d}@relay.internal" for i in range(n_a)])
    df["is_anomaly"] = [0]*n_n + [1]*n_a
    return run_model(df.set_index("sender"))


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("MAILSAFE AI")
    st.caption("Batch Behavioral Anomaly Profiling")
    st.divider()

    st.subheader("Upload Raw Email CSV")
    st.markdown(
        "Upload a **raw Enron-style CSV** (columns: `file`, `message`). "
        "The app parses the data and scores senders against the pre-trained Isolation Forest."
    )
    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    if uploaded:
        max_rows = st.slider("Max emails to parse", 5_000, 100_000, 50_000, step=5_000)
    st.divider()

    st.subheader("Detection Threshold")
    threshold = st.slider("Flag senders above score:", 0, 100, 65, step=5)
    st.divider()

    st.subheader("Search Sender")
    search_query = st.text_input("Email address", placeholder="user@example.com")


# ══════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════

st.title("MAILSAFE AI")
st.markdown("**Behavioral Anomaly Detection — Flags senders whose email patterns deviate significantly from the norm.**")
st.divider()

if uploaded is not None:
    # ── Run full pipeline ──────────────────────────────────
    pipeline_ph = st.empty()

    with pipeline_ph.container():
        with st.status("⚙️ Running pipeline…", expanded=True) as status:
            st.write("📥 Reading and parsing emails…")
            file_bytes = uploaded.read()
            parsed_df  = parse_raw_csv(file_bytes, max_emails=max_rows)

            if parsed_df.empty:
                st.error("❌ No valid emails could be parsed. Make sure your CSV has 'file' and 'message' columns.")
                st.stop()

            st.write(f"✅ Parsed **{len(parsed_df):,}** valid emails.")

            st.write("🔧 Building sender behavioural profiles…")
            profiles_df = build_profiles(parsed_df)
            st.write(f"✅ Profiled **{len(profiles_df):,}** unique senders.")

            st.write("🤖 Scoring against pre-trained Isolation Forest model…")
            results_df  = run_model(profiles_df)
            flagged_n   = int(results_df["is_anomaly"].sum())
            st.write(f"✅ Scoring complete — **{flagged_n}** senders flagged as anomalous.")
            status.update(label="✅ Pipeline complete!", state="complete", expanded=False)

    df = results_df.reset_index()      # 'sender' becomes a column

else:
    st.info("👈 No file uploaded — showing **demo data**. Upload a raw Enron CSV in the sidebar to run the real pipeline.", icon="ℹ️")
    df = demo_results().reset_index()


# ── Apply threshold ────────────────────────────────────────────────────────────
df["flagged"]    = df["anomaly_score"] >= threshold
flagged_df       = df[df["flagged"]].sort_values("anomaly_score", ascending=False)
clean_df         = df[~df["flagged"]]
df["Type"]       = df["flagged"].map({True: "🔴 Suspicious", False: "🟢 Normal"})


# ══════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════

c1, c2, c3, c4 = st.columns(4)
kpi_data = [
    (len(df),                                            "#e94560", "Total Senders Analyzed"),
    (len(flagged_df),                                    "#e94560", "Flagged as Suspicious"),
    (f"{100*len(flagged_df)/max(len(df),1):.1f}%",      "#f5a623", "Flag Rate"),
    (round(df["anomaly_score"].mean(), 1),               "#4caf50", "Avg Anomaly Score"),
]
for col, (val, color, label) in zip([c1, c2, c3, c4], kpi_data):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:{color}">{val}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CHARTS ROW 1: PCA scatter + Score histogram
# ══════════════════════════════════════════════════════════

col_a, col_b = st.columns([1.2, 1])

with col_a:
    st.subheader("Behavioral Cluster Map (PCA)")
    fig_scatter = px.scatter(
        df, x="pca_x", y="pca_y", color="Type",
        color_discrete_map={"Suspicious": "#e94560", "Normal": "#4caf50"},
        hover_data={"sender": True, "anomaly_score": True, "pca_x": False, "pca_y": False},
        size="anomaly_score", size_max=18, template="plotly_dark",
    )
    fig_scatter.update_layout(
        margin=dict(t=20, b=20), legend_title="Status",
        xaxis_title="PCA Component 1", yaxis_title="PCA Component 2",
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Normal senders (green) cluster tightly together in the centre, reflecting similar "
        "low-volume, business-hours behaviour. Anomalous senders (red) are projected outward "
        "into sparse regions because one or more of their behavioural features is an extreme outlier. "
        "Larger dot size indicates a higher anomaly score."
    )

with col_b:
    st.subheader("Anomaly Score Distribution")
    fig_hist = px.histogram(
        df, x="anomaly_score", nbins=30,
        color="Type",
        color_discrete_map={"Suspicious": "#e94560", "Normal": "#4caf50"},
        template="plotly_dark",
        barmode="overlay",
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="white",
                       annotation_text=f"Threshold ({threshold})", annotation_position="top right")
    fig_hist.update_layout(
        margin=dict(t=20, b=20),
        xaxis_title="Anomaly Score", yaxis_title="Count",
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    pct_flagged = round(100 * len(flagged_df) / max(len(df), 1), 1)
    st.caption(
        f"The majority of senders score below 25, indicating normal human email rhythms. "
        f"A distinct high-score tail (above the dashed threshold line at {threshold}) contains "
        f"{len(flagged_df)} senders ({pct_flagged}%) identified as anomalous by the Isolation Forest. "
        "Move the threshold slider in the sidebar to adjust sensitivity."
    )


# ══════════════════════════════════════════════════════════
# CHARTS ROW 2: Box plots
# ══════════════════════════════════════════════════════════

col_c, col_d = st.columns(2)
_type_map = {True: "Suspicious", False: "Normal"}

_box_inferences = {
    "night_email_ratio": (
        "Night Email Ratio — Normal vs Suspicious",
        "Anomalous senders show a significantly higher median night-email ratio, "
        "a strong indicator of automated cron-job or bot activity operating outside business hours. "
        "Most legitimate employees send less than 10% of their emails at night."
    ),
    "burst_score": (
        "Burst Score (Std Dev) — Normal vs Suspicious",
        "The burst score measures how unevenly emails are distributed across hourly time bins. "
        "Anomalous senders have a wider spread and higher median, indicating sudden high-volume "
        "sending surges characteristic of batch-send scripts and mailing-list servers."
    ),
}

for col, feat in [(col_c, "night_email_ratio"), (col_d, "burst_score")]:
    title, inference = _box_inferences[feat]
    with col:
        st.subheader(title)
        fig_box = px.box(
            df, x=df["flagged"].map(_type_map), y=feat,
            color=df["flagged"].map(_type_map),
            color_discrete_map={"Suspicious": "#e94560", "Normal": "#4caf50"},
            template="plotly_dark", points="all",
        )
        fig_box.update_layout(
            margin=dict(t=20, b=20), showlegend=False,
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            xaxis_title="", yaxis_title=feat,
        )
        col.plotly_chart(fig_box, use_container_width=True)
        col.caption(inference)


# ══════════════════════════════════════════════════════════
# RADAR CHART — top suspicious vs avg normal
# ══════════════════════════════════════════════════════════

st.subheader("Behavioral Radar — Top Suspicious Sender vs Average Normal Sender")

if len(flagged_df) > 0:
    top_row      = flagged_df.iloc[0]
    normal_means = clean_df[FEATURE_COLS].mean()
    combined     = df[FEATURE_COLS]
    f_min, f_max = combined.min(), combined.max()
    norm         = lambda row: ((row - f_min) / (f_max - f_min + 1e-9)).clip(0, 1)
    top_n        = norm(top_row[FEATURE_COLS])
    avg_n        = norm(normal_means)
    labels       = [c.replace("_", " ").title() for c in FEATURE_COLS]

    fig_radar = go.Figure()
    for vals, name, color, fill in [
        (top_n, top_row['sender'][:45], "#e94560", "rgba(233,69,96,0.2)"),
        (avg_n, "Avg Normal Sender",    "#4caf50", "rgba(76,175,80,0.15)"),
    ]:
        fig_radar.add_trace(go.Scatterpolar(
            r=list(vals) + [vals.iloc[0]], theta=labels + [labels[0]],
            fill="toself", name=name, line_color=color, fillcolor=fill,
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor="#1e1e2e"), paper_bgcolor="#1e1e2e",
        font_color="white", legend=dict(orientation="h", y=-0.1),
        margin=dict(t=30, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    dom_feat = top_n.idxmax().replace("_", " ")
    st.caption(
        f"The radar chart overlays the highest-scoring flagged sender against the average profile of all "
        f"normal senders across all 10 behavioural dimensions. The suspicious sender's polygon (red) "
        f"extends far beyond the normal sender (green) particularly on '{dom_feat}', "
        "confirming which specific behavioural feature is driving the anomaly detection decision."
    )


# ══════════════════════════════════════════════════════════
# FLAGGED SENDERS TABLE
# ══════════════════════════════════════════════════════════

st.subheader(f"Flagged Senders (Score ≥ {threshold})")

def risk_label(s):
    return "High" if s >= 70 else ("Medium" if s >= 40 else "Low")

if len(flagged_df) > 0:
    show = flagged_df[["sender","anomaly_score","total_emails_sent",
                        "emails_per_day","night_email_ratio",
                        "unique_recipients","burst_score"]].copy()
    show["Risk"] = show["anomaly_score"].apply(risk_label)
    show["night_email_ratio"] = (show["night_email_ratio"] * 100).round(1).astype(str) + "%"
    show.columns = ["Sender","Score","Emails Sent","Per Day","Night %","Unique Recips","Burst","Risk"]
    st.dataframe(show.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.caption(
        f"Table shows all {len(flagged_df)} senders whose normalised anomaly score exceeds the "
        f"current threshold of {threshold}. Senders with high Night % and Burst values are most "
        "likely automated services or compromised accounts operating outside normal business hours. "
        "Adjust the threshold slider in the sidebar to widen or narrow the flagged set."
    )
else:
    st.info("No senders flagged at this threshold. Try lowering it in the sidebar.")


# ══════════════════════════════════════════════════════════
# SENDER SEARCH
# ══════════════════════════════════════════════════════════

if search_query:
    st.subheader(f"Sender Profile: {search_query}")
    match = df[df["sender"].str.contains(search_query, case=False, na=False)]
    if not match.empty:
        row = match.iloc[0]
        score = row["anomaly_score"]
        st.markdown(f"**Anomaly Score:** `{score}` &nbsp; {risk_label(score)}")
        profile_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Value":   [round(row[c], 4) for c in FEATURE_COLS],
        })
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Sender not found in dataset.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("MAILSAFE AI · Batch Behavioral Anomaly Profiling · Isolation Forest + Streamlit")
