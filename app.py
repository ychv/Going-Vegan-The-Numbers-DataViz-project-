import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import lsq_linear

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Going Vegan : The Numbers",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Crimson Pro', Georgia, serif; }
  .main { background-color: #0f0b06; }
  .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1500px; }

  /* Hide sidebar completely */
  section[data-testid="stSidebar"],
  button[data-testid="collapsedControl"] { display: none !important; }

  .section-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #6b5538;
    margin-bottom: 0.5rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a1f10;
  }
  h1 { color: #f0e4c8 !important; font-weight: 300 !important; letter-spacing: -1px !important; }
  h2, h3 { color: #e8d5b0 !important; font-weight: 400 !important; }
  p, li { color: #b09a78 !important; }

  /* Number inputs */
  div[data-testid="stNumberInput"] input {
    background: #1a1208 !important;
    border: 1px solid #3d2e18 !important;
    color: #e8d5b0 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
  }
  div[data-testid="stNumberInput"] input:focus { border-color: #C9973A !important; }
  div[data-testid="stNumberInput"] button {
    background: #2a1f10 !important;
    border-color: #3d2e18 !important;
    color: #9a7e5a !important;
  }

  /* Metric cards */
  div[data-testid="stMetric"] {
    background: #1a1208; border: 1px solid #2a1f10;
    border-radius: 10px; padding: 12px 16px;
  }
  div[data-testid="stMetric"] label {
    color: #9a7e5a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
  }
  div[data-testid="stMetric"] div { color: #C9973A !important; }

  /* Buttons */
  .stButton > button {
    background: #2a1f10 !important; border: 1px solid #3d2e18 !important;
    color: #e8d5b0 !important; border-radius: 8px !important;
    font-family: 'Crimson Pro', serif !important; transition: all .2s !important;
  }
  .stButton > button:hover { border-color: #C9973A !important; color: #C9973A !important; }

  /* Expanders — styled as food group sections */
  div[data-testid="stExpander"] {
    background: #1a1208 !important;
    border: 1px solid #2a1f10 !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
  }
  div[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #9a7e5a !important;
    padding: 8px 12px !important;
  }
  div[data-testid="stExpander"] summary:hover { color: #C9973A !important; }
  div[data-testid="stExpander"] summary svg { color: #6b5538 !important; }

  /* Toggle */
  div[data-testid="stToggle"] label { color: #9a7e5a !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("3_data_v2.csv")
    df.columns = df.columns.str.strip()
    df = df[
        (df["LIBGR_DIDIT_eng"] != "Water and beverages") &
        (df["LIBGR_DIDIT_eng"] != "Ready meal and sandwich") &
        (df["LIBGR_DIDIT_eng"] != "Sugary food")
    ]
    id_cols = ["LIBGR_DIDIT_eng", "LIBSGR_DIDIT_eng", "LIBFAM_DIDIT_eng", "pop"]
    drop_cols = ["genre", "Code_EAT2", "Libell__eat2", "Libell__eat2_eng",
                 "LIBGR_DIDIT", "LIBSGR_DIDIT", "LIBFAM_DIDIT"]
    num_cols = list(df.columns.difference(id_cols + drop_cols))
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.groupby(id_cols, as_index=False)[num_cols].mean()
    # Composite key to avoid name collisions across groups (e.g. "Vegetable fat")
    df["fam_key"] = df["LIBGR_DIDIT_eng"] + "|" + df["LIBFAM_DIDIT_eng"]
    return df

df = load_data()

# Use a single reference population for nutrient coefficients
# (pop only affects obs/OPT, not the per-100g nutritional values which are averaged anyway)
REF_POP = df["pop"].iloc[0]
pop_df  = df[df["pop"] == REF_POP].copy()

# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
GROUP_COLORS = {
    "Meat/fish/egg":            "#B5553A",
    "Fruits and vegetables":    "#5A8A5E",
    "Starchy food":             "#C9973A",
    "Dairy product":            "#7A9EC2",
    "Fats":                     "#D4A853",
    "Plant-based alternative":  "#6BAA8E",
}
SUBGROUP_COLORS = {
    "Meat":                      "#B5553A",
    "Fish":                      "#3E7EA6",
    "Egg":                       "#E0A84B",
    "Cheese":                    "#7A9EC2",
    "Milk":                      "#9AC0D4",
    "Yoghurt":                   "#5A8AA0",
    "Starchy food, refined":     "#C9973A",
    "Starchy food, unrefined":   "#A07828",
    "Vegetable, soup":           "#5A8A5E",
    "Fresh and processed fruits":"#7AB05A",
    "Plant based proteins":      "#6BAA8E",
    "Vegetable fat":             "#D4A853",
    "Animal fat":                "#B08A5A",
}

# ── NUTRIENT COLUMNS (used for scenario optimisation) ─────────────────────────
NUTRIENT_COLS = [
    "PROT_DIG", "FIBRES", "GLUCIDES", "LIPIDES",
    "MIN_NA", "MIN_MG", "MIN_P", "MIN_K", "MIN_CA", "MIN_FE",
    "MIN_CU", "MIN_ZN", "MIN_SE", "MIN_I",
    "VIT_A", "VIT_D", "VIT_E", "VIT_C",
    "VIT_B1", "VIT_B2", "VIT_B3", "VIT_B5", "VIT_B6", "VIT_B9", "VIT_B12",
    "TRP_dig", "THR_dig", "ILE_dig", "LEU_dig", "LYS_dig",
    "MET_dig", "CYS_dig", "PHE_dig", "TYR_dig", "VAL_dig", "HIS_dig",
]

# ── BUILD FAMILY CATALOGUE (one row per unique fam_key) ───────────────────────
# Base cols first, then nutrient cols — deduplication ensures no repeated column
_base_cols = ["fam_key", "LIBFAM_DIDIT_eng", "LIBGR_DIDIT_eng", "LIBSGR_DIDIT_eng",
              "prix_pond", "ENERKC", "Climate_Change", "Water_Consumption",
              "Land_competition", "Cumulative_Energy_Demand", "Biodiversity", "obs"]
_all_cols = _base_cols + [c for c in NUTRIENT_COLS if c not in _base_cols]

family_catalogue = (
    pop_df[_all_cols]
    .drop_duplicates(subset="fam_key")
    .sort_values(["LIBGR_DIDIT_eng", "LIBFAM_DIDIT_eng"])
    .reset_index(drop=True)
)
all_keys = family_catalogue["fam_key"].tolist()

# ── INIT SESSION STATE ────────────────────────────────────────────────────────
if "qty_dict" not in st.session_state:
    st.session_state["qty_dict"] = {k: 0 for k in all_keys}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def compute_plate(qty_dict):
    """Compute plate metrics from {fam_key: grams} dict."""
    d = family_catalogue.copy()
    d["qty_g"] = d["fam_key"].map(qty_dict).fillna(0)
    d = d[d["qty_g"] > 0].copy()
    d["cost"]   = d["qty_g"] * d["prix_pond"] / 100
    d["prot_g"] = d["qty_g"] * d["PROT_DIG"]  / 100
    d["kcal"]   = d["qty_g"] * d["ENERKC"]    / 100
    for env in ["Climate_Change", "Water_Consumption", "Land_competition",
                "Cumulative_Energy_Demand", "Biodiversity"]:
        d[f"env_{env}"] = d["qty_g"] * d[env] / 100
    return d

def make_pie(plate_df, group_col="LIBGR_DIDIT_eng", colors=GROUP_COLORS, height=300):
    grouped = plate_df.groupby(group_col)["qty_g"].sum().reset_index()
    grouped = grouped[grouped["qty_g"] > 0]
    grouped["color"] = grouped[group_col].map(colors).fillna("#888")
    fig = go.Figure(go.Pie(
        labels=grouped[group_col], values=grouped["qty_g"],
        marker=dict(colors=grouped["color"].tolist(),
                    line=dict(color="#0f0b06", width=2)),
        textinfo="percent", textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f} g (%{percent})<extra></extra>",
        hole=0.25,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False, height=height,
    )
    return fig

def generate_alternative(plate_df, exclude_subgroups):
    """
    Build a nutritionally equivalent plate without the excluded subgroups.
    Uses bounded least-squares (lsq_linear) to find quantities of remaining
    foods that best match the original plate's nutrient profile
    (digestible proteins, amino acids, minerals, vitamins, macros).
    """
    if not exclude_subgroups:
        return plate_df.copy()

    excl = set(exclude_subgroups)

    # ── 1. Nutrient target from current plate (use .values to avoid index issues)
    nut_matrix = plate_df[NUTRIENT_COLS].values          # (n_foods, n_nutrients)
    quantities = plate_df["qty_g"].values[:, None]        # (n_foods, 1)
    target = (nut_matrix * quantities / 100.0).sum(axis=0)  # (n_nutrients,)

    # If plate is empty, just filter
    if target.sum() == 0:
        return plate_df[~plate_df["LIBSGR_DIDIT_eng"].isin(excl)].copy()

    # ── 2. Candidate foods: full catalogue minus excluded subgroups ────────────
    candidates = (
        family_catalogue[~family_catalogue["LIBSGR_DIDIT_eng"].isin(excl)]
        .copy()
        .reset_index(drop=True)
    )
    if candidates.empty:
        return pd.DataFrame(columns=plate_df.columns)

    # ── 3. Build normalised system A·q ≈ b ────────────────────────────────────
    # A: (n_nutrients × n_foods), per gram
    A = candidates[NUTRIENT_COLS].values.T / 100.0
    b = target

    # Normalise each nutrient so vitamins weigh as much as macros
    scales = np.where(b > 1e-9, b, 1.0)
    A_norm = A / scales[:, None]
    b_norm = b / scales

    # ── 4. Solve with per-food upper bound of 500g ─────────────────────────────
    res = lsq_linear(A_norm, b_norm, bounds=(0.0, 500.0), method="bvls")
    q = res.x

    # ── 5. Assemble result dataframe ───────────────────────────────────────────
    result = candidates.copy()
    result["qty_g"] = q
    result = result[result["qty_g"] > 0.5].copy()

    result["cost"]   = result["qty_g"] * result["prix_pond"] / 100
    result["prot_g"] = result["qty_g"] * result["PROT_DIG"]  / 100
    result["kcal"]   = result["qty_g"] * result["ENERKC"]    / 100
    for env in ["Climate_Change", "Water_Consumption", "Land_competition",
                "Cumulative_Energy_Demand", "Biodiversity"]:
        result[f"env_{env}"] = result["qty_g"] * result[env] / 100

    return result

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;margin-bottom:0.5rem;padding-top:1.5rem">'
    '<div style="margin:0;font-size:2rem;font-weight:300;letter-spacing:-1px;color:#f0e4c8;font-family:\'Crimson Pro\',Georgia,serif">Going Vegan : the Numbers to help you decide</div>'
    '</div>',
    unsafe_allow_html=True
)

# Controls row — below title, no risk of being clipped
ctrl_l, ctrl_c, ctrl_r = st.columns([1, 2, 1])
with ctrl_l:
    show_subgroup = st.toggle("Sub-group detail", value=False)
with ctrl_c:
    pass  # spacer
with ctrl_r:
    if st.button("↺ Reset to 0g", use_container_width=True):
        for k in all_keys:
            st.session_state[f"inp__{k}"] = 0.0
        st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── COMPUTE CURRENT PLATE ─────────────────────────────────────────────────────
base_plate = compute_plate(st.session_state["qty_dict"])

# ── KPI ROW ───────────────────────────────────────────────────────────────────
total_g    = base_plate["qty_g"].sum()
total_kcal = base_plate["kcal"].sum()
total_prot = base_plate["prot_g"].sum()
total_cost = base_plate["cost"].sum()
total_co2  = base_plate["env_Climate_Change"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("⚖️ Quantity", f"{total_g:.0f} g")
k2.metric("🔥 Kcal",     f"{total_kcal:.0f} kcal")
k3.metric("💪 Proteins", f"{total_prot:.1f} g")
k4.metric("💶 Price",    f"{total_cost:.2f} €")
k5.metric("☁️ CO₂",      f"{total_co2:.2f} kg")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── THREE-COLUMN LAYOUT ───────────────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1.0, 1.8, 1.4])

# ── LEFT: legend ──────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">Legend</div>', unsafe_allow_html=True)
    grp_col   = "LIBSGR_DIDIT_eng" if show_subgroup else "LIBGR_DIDIT_eng"
    color_map = SUBGROUP_COLORS     if show_subgroup else GROUP_COLORS

    groups_in_data = (
        base_plate.groupby(grp_col)["qty_g"].sum()
        .sort_values(ascending=False)
    )
    groups_in_data = groups_in_data[groups_in_data > 0]

    for grp_name, grp_g in groups_in_data.items():
        pct = grp_g / total_g * 100 if total_g > 0 else 0
        c   = color_map.get(grp_name, "#888")
        st.markdown(
            f'<div style="background:#1a1208;border:1px solid #2a1f10;border-radius:8px;'
            f'padding:7px 12px;margin:4px 0;display:flex;align-items:center;gap:8px">'
            f'<div style="width:10px;height:10px;border-radius:2px;background:{c};flex-shrink:0"></div>'
            f'<span style="font-size:13px;color:#b09a78;flex:1">{grp_name}</span>'
            f'<span style="font-size:11px;color:{c};font-family:\'DM Mono\',monospace">'
            f'{pct:.0f}%</span></div>',
            unsafe_allow_html=True
        )

# ── CENTER: pie ───────────────────────────────────────────────────────────────
with col_center:
    st.markdown('<div class="section-title">Current plate</div>', unsafe_allow_html=True)
    pie_colors = SUBGROUP_COLORS if show_subgroup else GROUP_COLORS
    st.plotly_chart(
        make_pie(base_plate, grp_col, colors=pie_colors),
        use_container_width=True, config={"displayModeBar": False}
    )
    st.markdown(
        '<div style="text-align:center;font-family:\'DM Mono\',monospace;font-size:10px;'
        'letter-spacing:3px;color:#6b5538;margin-top:-12px">'
        '── EDIT QUANTITIES ON THE RIGHT ──</div>',
        unsafe_allow_html=True
    )

# ── RIGHT: plate editor with expanders ────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">Build your plate (g)</div>', unsafe_allow_html=True)

    current_qty = st.session_state["qty_dict"]

    for grp_name, grp_color in GROUP_COLORS.items():
        fams = family_catalogue[family_catalogue["LIBGR_DIDIT_eng"] == grp_name]
        if fams.empty:
            continue

        # Count how many families have non-zero qty for the expander label
        nonzero = sum(1 for _, r in fams.iterrows() if current_qty.get(r["fam_key"], 0) > 0)
        total_grp_g = sum(current_qty.get(r["fam_key"], 0) for _, r in fams.iterrows())
        label = f"{grp_name}  ·  {total_grp_g:.0f} g"

        with st.expander(label, expanded=False):
            # Colored group indicator inside
            st.markdown(
                f'<div style="width:100%;height:2px;background:{grp_color};'
                f'border-radius:1px;margin-bottom:8px"></div>',
                unsafe_allow_html=True
            )
            for _, fam_row in fams.iterrows():
                key     = fam_row["fam_key"]
                fam     = fam_row["LIBFAM_DIDIT_eng"]
                display = fam if len(fam) <= 28 else fam[:26] + "…"

                name_col, input_col = st.columns([2, 1])
                with name_col:
                    st.markdown(
                        f'<div style="color:#9a7e5a;font-size:12px;padding-top:7px;'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                        f'{display}</div>',
                        unsafe_allow_html=True
                    )
                with input_col:
                    new_val = st.number_input(
                        label=key,
                        min_value=0.0, max_value=2000.0,
                        value=float(current_qty.get(key, 0)),
                        step=10.0, format="%.0f",
                        label_visibility="collapsed",
                        key=f"inp__{key}",
                    )
                    if new_val != current_qty.get(key, 0):
                        st.session_state["qty_dict"][key] = new_val
                        st.rerun()

# ── DIVIDER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin:2rem 0 1.5rem">
  <div style="flex:1;height:1px;background:#2a1f10"></div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:4px;
              color:#6b5538;text-transform:uppercase">⇣ Nutritional equivalent plates</div>
  <div style="flex:1;height:1px;background:#2a1f10"></div>
</div>
""", unsafe_allow_html=True)

# ── ALTERNATIVES ──────────────────────────────────────────────────────────────
scenarios = {
    "🥩 Current plate": {"exclude": [],                 "color": "#B5553A", "desc": "Current composition"},
    "🐟 No meat":        {"exclude": ["Meat"],           "color": "#3E7EA6", "desc": "Meat excluded"},
    "🥚 No fish":        {"exclude": ["Fish", "Meat"],   "color": "#E0A84B", "desc": "Meat & fish excluded"},
    "🌱 Full vegan":     {"exclude": ["Meat", "Fish", "Egg", "Cheese", "Milk",
                                       "Yoghurt", "Animal fat", "Milk based dessert"],
                          "color": "#6BAA8E", "desc": "All animal products excluded"},
}

alt_cols = st.columns(4)
for i, (scenario_name, scenario_info) in enumerate(scenarios.items()):
    with alt_cols[i]:
        alt_df = generate_alternative(base_plate, scenario_info["exclude"])
        color  = scenario_info["color"]

        if alt_df.empty:
            st.warning(f"No data for: {scenario_name}")
            continue

        # ── Self-contained header block (no unclosed divs) ──
        st.markdown(
            f'<div style="border-top:3px solid {color};border-radius:4px 4px 0 0;'
            f'background:#1a1208;border-left:1px solid #2a1f10;border-right:1px solid #2a1f10;'
            f'padding:14px 14px 10px 14px;margin-bottom:0">'
            f'<div style="color:{color};font-size:15px;font-weight:600;margin-bottom:3px">'
            f'{scenario_name}</div>'
            f'<div style="color:#6b5538;font-size:11px;font-family:\'DM Mono\',monospace;'
            f'letter-spacing:1px">{scenario_info["desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Pie ──
        fig_mini = make_pie(alt_df, "LIBGR_DIDIT_eng", colors=GROUP_COLORS, height=190)
        fig_mini.update_layout(margin=dict(t=4, b=4, l=4, r=4))
        st.plotly_chart(fig_mini, use_container_width=True, config={"displayModeBar": False})

        # ── Metrics ──
        m1, m2 = st.columns(2)
        m1.metric("💶 Price",    f"{alt_df['cost'].sum():.2f} €")
        m2.metric("💪 Proteins", f"{alt_df['prot_g'].sum():.1f} g")
        st.metric("⚖️ Quantity",  f"{alt_df['qty_g'].sum():.0f} g")

# ── ENV COMPARISON CHART ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Environmental impact comparison</div>',
            unsafe_allow_html=True)

env_metrics = {
    "Climate_Change":           ("☁️ CO₂ (kg eq.)",  1),
    "Water_Consumption":        ("💧 Water (L)",      1),
    "Land_competition":         ("🌱 Land use (m²)",  1),
    "Cumulative_Energy_Demand": ("⚡ Energy (10 MJ)", 0.1),
}

fig_comp = go.Figure()
for env_col, (env_label, scale) in env_metrics.items():
    y_vals = []
    for scenario_info in scenarios.values():
        alt_df = generate_alternative(base_plate, scenario_info["exclude"])
        val = alt_df[f"env_{env_col}"].sum() * scale if not alt_df.empty else 0
        y_vals.append(val)
    fig_comp.add_trace(go.Bar(
        name=env_label, x=list(scenarios.keys()), y=y_vals,
        hovertemplate="%{x}<br>" + env_label + ": <b>%{y:.3f}</b><extra></extra>",
    ))

fig_comp.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    barmode="group", height=300,
    margin=dict(t=20, b=20, l=20, r=20),
    legend=dict(font=dict(color="#9a7e5a", size=12), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(color="#9a7e5a", tickfont=dict(size=12), gridcolor="#1a1208"),
    yaxis=dict(color="#9a7e5a", tickfont=dict(size=11), gridcolor="#2a1f10"),
    colorway=["#B5553A", "#3E7EA6", "#6BAA8E", "#C9973A"],
)
st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2rem;padding-top:1rem;border-top:1px solid #1a1208">
  <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;color:#6b5538">
    DATA : CIQUAL · INCA2 · AGRIBALYSE V3
  </span>
</div>
""", unsafe_allow_html=True)
