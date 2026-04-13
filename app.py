import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import lsq_linear

####################### PAGE CONFIG #######################
st.set_page_config(
    page_title="Going Vegan : The Numbers",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

######################### STYLES ##########################
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Crimson Pro', Georgia, serif; }
  .main { background-color: #0f0b06; }
  .block-container { padding: 3rem 2rem 2rem 2rem; max-width: 1500px; }

  section[data-testid="stSidebar"],
  button[data-testid="collapsedControl"] { display: none !important; }

  .section-title {
    font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 4px;
    text-transform: uppercase; color: #6b5538; margin-bottom: 0.5rem;
    padding-bottom: 6px; border-bottom: 1px solid #2a1f10;
  }
  h1 { color: #f0e4c8 !important; font-weight: 300 !important; letter-spacing: -1px !important; }
  h2, h3 { color: #e8d5b0 !important; font-weight: 400 !important; }
  p, li { color: #b09a78 !important; }

  div[data-testid="stNumberInput"] input {
    background: #1a1208 !important; border: 1px solid #3d2e18 !important;
    color: #e8d5b0 !important; border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important; font-size: 12px !important;
  }
  div[data-testid="stNumberInput"] input:focus { border-color: #C9973A !important; }
  /* Hide +/- stepper buttons */
  div[data-testid="stNumberInput"] button { display: none !important; }
  div[data-testid="stNumberInput"] > div { gap: 0 !important; }
  div[data-testid="stNumberInput"] input { width: 100% !important; }

  div[data-testid="stMetric"] {
    background: #1a1208; border: 1px solid #2a1f10;
    border-radius: 10px; padding: 12px 16px;
  }
  div[data-testid="stMetric"] label {
    color: #9a7e5a !important; font-family: 'DM Mono', monospace !important; font-size: 11px !important;
  }
  div[data-testid="stMetric"] div { color: #C9973A !important; }

  .stButton > button {
    background: #2a1f10 !important; border: 1px solid #3d2e18 !important;
    color: #e8d5b0 !important; border-radius: 8px !important;
    font-family: 'Crimson Pro', serif !important; transition: all .2s !important;
  }
  .stButton > button:hover { border-color: #C9973A !important; color: #C9973A !important; }

  /* Validate button — gold accent */
  .stButton[data-testid="validate-btn"] > button {
    border-color: #C9973A !important; color: #C9973A !important;
  }

  div[data-testid="stExpander"] {
    background: #1a1208 !important; border: 1px solid #2a1f10 !important;
    border-radius: 10px !important; margin-bottom: 6px !important;
  }
  div[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: #9a7e5a !important; padding: 8px 12px !important;
  }
  div[data-testid="stExpander"] summary:hover { color: #C9973A !important; }
  div[data-testid="stExpander"] summary svg { color: #6b5538 !important; }

  div[data-testid="stToggle"] label { color: #9a7e5a !important; font-size: 13px !important; }

  /* Info banner */
  .info-banner {
    background: #120e06;
    border: 1px solid #2a1f10;
    border-left: 3px solid #6BAA8E;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 1.2rem;
    color: #9a7e5a;
    font-size: 13px;
    line-height: 1.6;
  }
  .info-banner b { color: #C9973A; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

###################### LOAD DATA ##########################
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
    df["fam_key"] = df["LIBGR_DIDIT_eng"] + "|" + df["LIBFAM_DIDIT_eng"]
    return df

df = load_data()
REF_POP = df["pop"].iloc[0]
pop_df  = df[df["pop"] == REF_POP].copy()

################### COLOR PALETTES ########################
GROUP_COLORS = {
    "Meat/fish/egg":"#B5553A",
    "Fruits and vegetables":"#5A8A5E",
    "Starchy food":"#C9973A",
    "Dairy product":"#7A9EC2",
    "Fats":"#D4A853",
    "Plant-based alternative":"#6BAA8E",
}
SUBGROUP_COLORS = {
    "Meat":"#B5553A", "Fish":"#3E7EA6",
    "Egg":"#E0A84B", "Cheese":"#7A9EC2",
    "Milk":"#9AC0D4", "Yoghurt":"#5A8AA0",
    "Starchy food, refined":"#C9973A", "Starchy food, unrefined":"#A07828",
    "Vegetable, soup":"#5A8A5E", "Fresh and processed fruits":"#7AB05A",
    "Plant based proteins":"#6BAA8E", "Vegetable fat":"#D4A853",
    "Animal fat":"#B08A5A",
}

##### NUTRIENT COLUMNS (FOR ALTERNATIVE COMPUTATION) ######
NUTRIENT_COLS = [
    "PROT_DIG", "FIBRES", "GLUCIDES", "LIPIDES",
    "MIN_NA", "MIN_MG", "MIN_P", "MIN_K", "MIN_CA", "MIN_FE",
    "MIN_CU", "MIN_ZN", "MIN_SE", "MIN_I",
    "VIT_A", "VIT_D", "VIT_E", "VIT_C",
    "VIT_B1", "VIT_B2", "VIT_B3", "VIT_B5", "VIT_B6", "VIT_B9", "VIT_B12",
    "TRP_dig", "THR_dig", "ILE_dig", "LEU_dig", "LYS_dig",
    "MET_dig", "CYS_dig", "PHE_dig", "TYR_dig", "VAL_dig", "HIS_dig",
]

# Priority: macros + digestible proteins + key amino acids + key minerals
_PRIMARY = {
    "PROT_DIG", "GLUCIDES", "LIPIDES", "FIBRES",
    "LYS_dig", "THR_dig", "LEU_dig", "ILE_dig", "VAL_dig",
    "MET_dig", "CYS_dig", "TRP_dig",
    "MIN_CA", "MIN_FE", "MIN_ZN", "MIN_MG",
}
# VIT_B12 tracked separately for warning (only in animal products)
NUTRIENT_WEIGHTS = np.array([
    3.0 if c in _PRIMARY else (0.05 if c == "VIT_B12" else 0.4)
    for c in NUTRIENT_COLS
])

################### FAMILY CATALOGUE ######################
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

#################### SAMPLE MEALS #########################
# fam_key: grams — mapped to actual catalogue keys
PREDEFINED_MEALS = [
    {
        "name": "Steak, vegetables & potatoes",
        "Meat/fish/egg|Red meat": 180,
        "Fruits and vegetables|Vegetable ": 180,
        "Starchy food|Potato": 200,
        "Fats|Vegetable oil": 15,
    },
    {
        "name": "Salmon & rice",
        "Meat/fish/egg|Fatty fish": 160,
        "Starchy food|Pasta, rice, semolina": 180,
        "Fruits and vegetables|Vegetable ": 80,
        "Fats|Vegetable oil": 10,
    },
    {
        "name": "Sausages & lentils",
        "Meat/fish/egg|Delicatessen": 140,
        "Starchy food|Dried vegetable": 220,
        "Fats|Vegetable oil": 10,
    },
    {
        "name": "Chicken, cream & mushrooms",
        "Meat/fish/egg|Poultry and venison": 180,
        "Fats|Cream": 80,
        "Fruits and vegetables|Vegetable ": 140,
        "Fats|Vegetable oil": 12,
    },
    {
        "name": "Bacon omelette",
        "Meat/fish/egg|Egg": 150,
        "Meat/fish/egg|Delicatessen": 70,
        "Fats|Butter": 15,
        "Fruits and vegetables|Vegetable ": 80,
    },
]

def init_meal(meal: dict):
    """
    Write a predefined meal into qty_dict and inp__ widget keys.
    """
    qty = {k: 0.0 for k in all_keys}
    for fk, g in meal.items():
        if fk == "name":
            continue
        if fk in qty:
            qty[fk] = float(g)
    st.session_state["qty_dict"] = qty
    for k in all_keys:
        st.session_state[f"inp__{k}"] = qty[k]

####################### SESSION INIT ######################
if "qty_dict" not in st.session_state:
    # Pick a random meal on first load
    meal = PREDEFINED_MEALS[np.random.randint(len(PREDEFINED_MEALS))]
    st.session_state["init_meal_name"] = meal["name"]
    init_meal(meal)
    st.session_state["pending_changes"] = False
else:
    # Sync inp__ → qty_dict on first build if inp__ keys exist
    for k in all_keys:
        if f"inp__{k}" not in st.session_state:
            st.session_state[f"inp__{k}"] = st.session_state["qty_dict"].get(k, 0.0)

if "pending_changes" not in st.session_state:
    st.session_state["pending_changes"] = False

################## HELPERS FUNCTIONS ######################
def compute_plate(qty_dict):
    """
    Compute total plate based on ingredient
    quantities dictionnary
    Returns : dataframe representing the plate
    """
    d = family_catalogue.copy()
    d["qty_g"] = d["fam_key"].map(qty_dict).fillna(0)
    d = d[d["qty_g"] > 0].copy()
    d["cost"]   = d["qty_g"] * d["prix_pond"] / 100
    d["prot_g"] = d["qty_g"] * d["PROT_DIG"]  / 100
    d["kcal"]   = d["qty_g"] * d["ENERKC"]    / 100
    d["Fe"]     = d["qty_g"] * d["MIN_FE"]    / 100
    for env in ["Climate_Change", "Water_Consumption", "Land_competition",
                "Cumulative_Energy_Demand", "Biodiversity"]:
        d[f"env_{env}"] = d["qty_g"] * d[env] / 100
    return d

def make_pie(plate_df, group_col="LIBGR_DIDIT_eng", colors=GROUP_COLORS, height=300):
    """
    Plot the plate as a pie chart
    Returns : pie chart figure
    """
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

## NUTRIENT WEIGHTS FOR ALTERNATIVE COMPUTATION
_PROTEIN_AA = {
    "PROT_DIG", "LYS_dig", "THR_dig", "LEU_dig", "ILE_dig",
    "VAL_dig", "MET_dig", "CYS_dig", "TRP_dig", "PHE_dig", "TYR_dig", "HIS_dig",
}
_MINERALS = {"MIN_CA", "MIN_FE", "MIN_ZN", "MIN_MG", "MIN_P"}
NUTRIENT_WEIGHTS = np.array([
    5.0 if c in _PROTEIN_AA else
     2.0 if c in {"FIBRES", "GLUCIDES", "LIPIDES"} else
     1.0 if c in _MINERALS else
     0.05 if c == "VIT_B12" else
     0.2
    for c in NUTRIENT_COLS
])

# Per-subgroup upper bounds (g) to prevents crazy stuff like 300g of sauce
SUBGROUP_MAX_G = {
    "Sauces and spices": 50,
    "Vegetable fat": 50,
    "Animal fat": 80,
    "Dry fruit and oleaginous": 100,
    "Fresh and processed fruits": 300,
    "Breakfast cereals": 150,
}
DEFAULT_MAX_G = 600.0


def generate_alternative(plate_df, exclude_subgroups):
    """
    Nutritionally equivalent plate using two-phase bounded least-squares.
    Prioritises matching digestible proteins + essential amino acids.
    Returns : result plate dataframe, b12_warning boolean.
    """
    if not exclude_subgroups:
        return plate_df.copy(), False

    excl = set(exclude_subgroups)

    # Nutrient target
    target = (
        plate_df[NUTRIENT_COLS].values *
        plate_df["qty_g"].values[:, None] / 100.0
    ).sum(axis=0)

    if target.sum() == 0:
        result = plate_df[~plate_df["LIBSGR_DIDIT_eng"].isin(excl)].copy()
        return result, False

    # Candidates + per-food bounds
    candidates = (
        family_catalogue[~family_catalogue["LIBSGR_DIDIT_eng"].isin(excl)]
        .copy().reset_index(drop=True)
    )
    if candidates.empty:
        return pd.DataFrame(columns=plate_df.columns), False

    ubs = np.array([
        SUBGROUP_MAX_G.get(row["LIBSGR_DIDIT_eng"], DEFAULT_MAX_G)
        for _, row in candidates.iterrows()
    ], dtype=float)

    # Normalised + weighted system
    A = candidates[NUTRIENT_COLS].values.T / 100.0   # (n_nut, n_foods)
    b = target
    scales = np.where(b > 1e-9, b, 1.0)
    w = NUTRIENT_WEIGHTS
    A_norm = (A / scales[:, None]) * w[:, None]
    b_norm = (b / scales) * w

    # Phase 1: full solve with tiny ridge
    alpha = 0.005
    n = A_norm.shape[1]
    A_aug = np.vstack([A_norm, alpha * np.eye(n)])
    b_aug = np.concatenate([b_norm, np.zeros(n)])
    res1 = lsq_linear(A_aug, b_aug, bounds=(np.zeros(n), ubs), method="bvls")
    q1 = res1.x

    # Phase 2: re-solve with top MAX_FOODS
    MAX_FOODS = 10
    top_idx = np.argsort(q1)[-MAX_FOODS:]
    A_sub = A_norm[:, top_idx]
    ubs_sub = ubs[top_idx]
    A_sub_aug = np.vstack([A_sub, alpha * np.eye(len(top_idx))])
    b_sub_aug = np.concatenate([b_norm, np.zeros(len(top_idx))])
    res2 = lsq_linear(A_sub_aug, b_sub_aug, bounds=(np.zeros(len(top_idx)), ubs_sub), method="bvls")
    q_final   = np.zeros(n)
    q_final[top_idx] = res2.x

    # Assemble
    result = candidates.copy()
    result["qty_g"] = q_final
    result = result[result["qty_g"] > 0.5].copy()

    result["cost"] = result["qty_g"] * result["prix_pond"] / 100
    result["prot_g"] = result["qty_g"] * result["PROT_DIG"]  / 100
    result["kcal"] = result["qty_g"] * result["ENERKC"]    / 100
    result["Fe"] = result["qty_g"] * result["MIN_FE"]    / 100
    for env in ["Climate_Change", "Water_Consumption", "Land_competition", "Cumulative_Energy_Demand", "Biodiversity"]:
        result[f"env_{env}"] = result["qty_g"] * result[env] / 100

    # B12 warning
    b12_idx = NUTRIENT_COLS.index("VIT_B12")
    achieved_b12 = (result[NUTRIENT_COLS].values * result["qty_g"].values[:, None] / 100).sum(axis=0)[b12_idx]
    b12_warning  = (target[b12_idx] > 0.1 and achieved_b12 / max(target[b12_idx], 1e-9) < 0.3)
    return result, b12_warning

################### ALTERNATIVES PLATES ###################
scenarios = {
    "🥩 Current plate": {"exclude": [], "color": "#B5553A", "desc": "Current composition"},
    "🐟 No meat": {"exclude": ["Meat"], "color": "#3E7EA6", "desc": "Meat excluded"},
    "🥚 No fish": {"exclude": ["Fish", "Meat"], "color": "#E0A84B", "desc": "Meat & fish excluded"},
    "🌱 Full vegan": {"exclude": ["Meat", "Fish", "Egg", "Cheese", "Milk", "Yoghurt", "Animal fat", "Milk based dessert"], 
                      "color": "#6BAA8E", "desc": "All animal products excluded"},
}

######################## HEADER ###########################
st.markdown(
    '<div style="text-align:center;margin-bottom:0.5rem">'
    '<div style="margin:0;font-size:2rem;font-weight:300;letter-spacing:-1px;color:#f0e4c8;'
    'font-family:\'Crimson Pro\',Georgia,serif">Going Vegan : the Numbers to help you decide</div>'
    '<div style="margin-top:0.6rem;font-size:1rem;font-weight:300;color:#9a7e5a;'
    'font-family:\'Crimson Pro\',Georgia,serif;font-style:italic;max-width:700px;'
    'margin-left:auto;margin-right:auto;line-height:1.6">'
    'This interactive dashboard is made for helping people see what a transition to a no-meat, '
    'vegetarian, or vegan diet would look like by finding more environmental and animal friendly '
    'alternatives to their favorite meals… without compromising on health ! Choose a sample meal or ' \
    'create your own using the editor on the right, and see the alternative below and their environmental impacts.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

ctrl_l, ctrl_c, ctrl_r = st.columns([1, 2, 1])
with ctrl_l:
    show_subgroup = st.toggle("Sub-group detail", value=False)
with ctrl_c:
    pass
with ctrl_r:
    r1, r2 = st.columns(2)
    with r1:
        if st.button("↺ Reset", use_container_width=True):
            for k in all_keys:
                st.session_state[f"inp__{k}"] = 0.0
            st.session_state["qty_dict"] = {k: 0.0 for k in all_keys}
            st.session_state["pending_changes"] = False
            st.rerun()
    with r2:
        meal_names = [m["name"] for m in PREDEFINED_MEALS]
        idx = st.session_state.get("init_meal_name", meal_names[0])
        chosen = st.selectbox("", meal_names,
                              index=meal_names.index(idx) if idx in meal_names else 0,
                              label_visibility="collapsed",
                              key="meal_select")
        if st.button("🍽 Load meal", use_container_width=True):
            meal = next(m for m in PREDEFINED_MEALS if m["name"] == chosen)
            st.session_state["init_meal_name"] = chosen
            init_meal(meal)
            st.session_state["pending_changes"] = False
            st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

######## COMPUTE PLATE (from committed qty_dict) ##########
base_plate = compute_plate(st.session_state["qty_dict"])

########################## KPI ROW ########################
total_g = base_plate["qty_g"].sum()
total_kcal = base_plate["kcal"].sum()
total_prot = base_plate["prot_g"].sum()
total_cost = base_plate["cost"].sum()
total_co2 = base_plate["env_Climate_Change"].sum()
total_Fe = base_plate["Fe"].sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("⚖️ Quantity", f"{total_g:.0f} g")
k2.metric("🔥 Kcal", f"{total_kcal:.0f} kcal")
k3.metric("💪 Proteins", f"{total_prot:.1f} g")
k4.metric("🦾 Iron", f"{total_Fe:.2f} mg")
k5.metric("💶 Price", f"{total_cost:.2f} €")
k6.metric("☁️ CO₂", f"{total_co2:.2f} kg")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

################### THREE-COLUMN LAYOUT ###################
col_left, col_center, col_right = st.columns([1.0, 1.8, 1.4])

###################### LEFT: legend #######################
with col_left:
    st.markdown('<div class="section-title">Legend</div>', unsafe_allow_html=True)
    grp_col = "LIBSGR_DIDIT_eng" if show_subgroup else "LIBGR_DIDIT_eng"
    color_map = SUBGROUP_COLORS if show_subgroup else GROUP_COLORS

    groups_in_data = base_plate.groupby(grp_col)["qty_g"].sum().sort_values(ascending=False)
    groups_in_data = groups_in_data[groups_in_data > 0]

    for grp_name, grp_g in groups_in_data.items():
        pct = grp_g / total_g * 100 if total_g > 0 else 0
        c = color_map.get(grp_name, "#888")
        st.markdown(
            f'<div style="background:#1a1208;border:1px solid #2a1f10;border-radius:8px;'
            f'padding:7px 12px;margin:4px 0;display:flex;align-items:center;gap:8px">'
            f'<div style="width:10px;height:10px;border-radius:2px;background:{c};flex-shrink:0"></div>'
            f'<span style="font-size:13px;color:#b09a78;flex:1">{grp_name}</span>'
            f'<span style="font-size:11px;color:{c};font-family:\'DM Mono\',monospace">'
            f'{pct:.0f}%</span></div>',
            unsafe_allow_html=True
        )

##################### CENTER : current plate ##############
with col_center:
    st.markdown('<div class="section-title">Current plate</div>', unsafe_allow_html=True)
    pie_colors = SUBGROUP_COLORS if show_subgroup else GROUP_COLORS
    st.plotly_chart(make_pie(base_plate, grp_col, colors=pie_colors),use_container_width=True, config={"displayModeBar": False}, key="pie_main")
    hint_color = "#C9973A" if st.session_state["pending_changes"] else "#6b5538"
    hint_text  = "── PENDING CHANGES — PRESS VALIDATE ──" if st.session_state["pending_changes"] \
                 else "── EDIT QUANTITIES ON THE RIGHT ──"
    st.markdown(
        f'<div style="text-align:center;font-family:\'DM Mono\',monospace;font-size:10px;'
        f'letter-spacing:3px;color:{hint_color};margin-top:-12px">{hint_text}</div>',
        unsafe_allow_html=True
    )

##################### RIGHT: plate editor #################
with col_right:
    # Validate button
    v_col, _ = st.columns([1, 1])
    with v_col:
        if st.button("✓ Validate quantities", use_container_width=True, key="validate_btn"):
            new_qty = {k: float(st.session_state.get(f"inp__{k}", 0.0)) for k in all_keys}
            st.session_state["qty_dict"] = new_qty
            st.session_state["pending_changes"] = False
            st.rerun()

    st.markdown('<div class="section-title" style="margin-top:8px">Build your plate (g)</div>',
                unsafe_allow_html=True)

    committed = st.session_state["qty_dict"]

    for grp_name, grp_color in GROUP_COLORS.items():
        fams = family_catalogue[family_catalogue["LIBGR_DIDIT_eng"] == grp_name]
        if fams.empty:
            continue

        total_grp_g = sum(committed.get(r["fam_key"], 0.0) for _, r in fams.iterrows())
        label = f"{grp_name}  ·  {total_grp_g:.0f} g"

        with st.expander(label, expanded=False):
            st.markdown(
                f'<div style="width:100%;height:2px;background:{grp_color};'
                f'border-radius:1px;margin-bottom:8px"></div>',
                unsafe_allow_html=True
            )
            for _, fam_row in fams.iterrows():
                fk = fam_row["fam_key"]
                fam = fam_row["LIBFAM_DIDIT_eng"]
                display = fam if len(fam) <= 28 else fam[:26] + "…"

                nc, ic = st.columns([2, 1])
                with nc:
                    st.markdown(
                        f'<div style="color:#9a7e5a;font-size:12px;padding-top:7px;'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                        f'{display}</div>',
                        unsafe_allow_html=True
                    )
                with ic:
                    prev = st.session_state.get(f"inp__{fk}", committed.get(fk, 0.0))
                    new_val = st.number_input(
                        label=fk, min_value=0.0, max_value=2000.0,
                        step=10.0, format="%.0f",
                        label_visibility="collapsed",
                        key=f"inp__{fk}",
                    )
                    # Detect pending changes (widget changed but not yet validated)
                    if abs(new_val - committed.get(fk, 0.0)) > 0.01:
                        st.session_state["pending_changes"] = True

####################### DIVIDER ###########################
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin:2rem 0 1rem">
  <div style="flex:1;height:1px;background:#2a1f10"></div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:4px;
              color:#6b5538;text-transform:uppercase">⇣ Nutritional equivalent plates</div>
  <div style="flex:1;height:1px;background:#2a1f10"></div>
</div>
""", unsafe_allow_html=True)

#################### INFO BANNER ##########################
st.markdown(
    '<div class="info-banner">'
    '<b>How alternatives are calculated</b> — Each alternative plate is optimised to '
    'match your meal\'s nutrient content as closely as possible, using bounded least-squares on '
    '<b>36 nutritional dimensions</b>: digestible proteins, macronutrients (carbohydrates, '
    'lipids, fibre), essential amino acids (Lys, Thr, Leu, Ile, Val, Met, Cys, Trp…), '
    'key minerals (Fe, Ca, Zn, Mg) and vitamins. '
    'Priority is given to proteins and amino acids over trace micronutrients. '
    'At most 7 food families are selected as alternatives for readability.'
    '</div>',
    unsafe_allow_html=True
)

################# COMPUTE ALL SCENARIOS ###################
scenario_results = {}
for sname, sinfo in scenarios.items():
    alt_df, b12_warn = generate_alternative(base_plate, sinfo["exclude"])
    scenario_results[sname] = {"df": alt_df, "b12_warn": b12_warn, **sinfo}

#################### ALTERNATIVE PLATES ###################
alt_cols = st.columns(4)
for i, (scenario_name, sdata) in enumerate(scenario_results.items()):
    with alt_cols[i]:
        alt_df = sdata["df"]
        color  = sdata["color"]

        if alt_df.empty:
            st.warning(f"No data for: {scenario_name}")
            continue

        st.markdown(
            f'<div style="border-top:3px solid {color};border-radius:4px 4px 0 0;'
            f'background:#1a1208;border-left:1px solid #2a1f10;border-right:1px solid #2a1f10;'
            f'padding:14px 14px 10px 14px;margin-bottom:0">'
            f'<div style="color:{color};font-size:15px;font-weight:600;margin-bottom:3px">'
            f'{scenario_name}</div>'
            f'<div style="color:#6b5538;font-size:11px;font-family:\'DM Mono\',monospace;'
            f'letter-spacing:1px">{sdata["desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        fig_mini = make_pie(alt_df, "LIBGR_DIDIT_eng", colors=GROUP_COLORS, height=190)
        fig_mini.update_layout(margin=dict(t=4, b=4, l=4, r=4))
        st.plotly_chart(fig_mini, use_container_width=True, config={"displayModeBar": False}, key=f"pie_scenario_{i}")

        m1, m2 = st.columns(2)
        m1.metric("💶 Price", f"{alt_df['cost'].sum():.2f} €")
        m2.metric("💪 Proteins", f"{alt_df['prot_g'].sum():.1f} g")
        m1.metric("⚖️ Quantity", f"{alt_df['qty_g'].sum():.0f} g")
        m2.metric("🦾 Iron", f"{alt_df['Fe'].sum():.2f} mg")

        if sdata.get("b12_warn"):
            st.markdown(
                '<div style="background:#1a0e06;border:1px solid #7A4A2A;border-radius:6px;'
                'padding:8px 10px;margin-top:6px;font-size:11px;color:#C9733A">'
                '⚠️ Low vitamin B12 — consider supplements'
                '</div>',
                unsafe_allow_html=True
            )

        # Food list
        if not alt_df.empty:
            foods_sorted = alt_df.sort_values("qty_g", ascending=False)
            with st.expander(f"{len(foods_sorted)} ingredients selected", expanded=False):
                for _, row in foods_sorted.iterrows():
                    c = GROUP_COLORS.get(row["LIBGR_DIDIT_eng"], "#888")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;'
                        f'padding:4px 0;border-bottom:1px solid #1a1208">'
                        f'<div style="width:6px;height:6px;border-radius:50%;'
                        f'background:{c};flex-shrink:0"></div>'
                        f'<span style="color:#b09a78;font-size:12px;flex:1">'
                        f'{row["LIBFAM_DIDIT_eng"]}</span>'
                        f'<span style="color:#6b5538;font-size:11px;'
                        f'font-family:\'DM Mono\',monospace">{row["qty_g"]:.0f}g</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

################ ENV IMPACTS COMPARISON ###################
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Environmental impact comparison</div>',
            unsafe_allow_html=True)

env_metrics = {
    "Climate_Change": ("☁️ CO₂ eq.", "kg"),
    "Water_Consumption": ("💧 Water",   "L"),
    "Land_competition": ("🌱 Land use","m²"),
    "Cumulative_Energy_Demand": ("⚡ Energy cost",  "MJ"),
}
env_scales = {
    "Climate_Change": 1, "Water_Consumption": 1,
    "Land_competition": 1, "Cumulative_Energy_Demand": 0.1,
}

env_cols = st.columns(4)
for col_idx, (env_col, (env_label, env_unit)) in enumerate(env_metrics.items()):
    scale = env_scales[env_col]
    x_labels = list(scenario_results.keys())
    y_vals = []
    colors_ = []
    for sname, sdata in scenario_results.items():
        adf = sdata["df"]
        val = adf[f"env_{env_col}"].sum() * scale if not adf.empty else 0
        y_vals.append(val)
        colors_.append(sdata["color"])

    fig = go.Figure(go.Bar(
        x=x_labels, y=y_vals,
        marker=dict(color=colors_, line=dict(color="#0f0b06", width=1)),
        hovertemplate="%{x}<br>" + env_label + ": <b>%{y:.2f} " + env_unit + "</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{env_label} ({env_unit})", font=dict(color="#9a7e5a", size=12), x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(color="#9a7e5a", tickfont=dict(size=9), gridcolor="#1a1208", tickangle=-20),
        yaxis=dict(color="#9a7e5a", tickfont=dict(size=10), gridcolor="#2a1f10"),
        showlegend=False,
    )
    with env_cols[col_idx]:
        st.plotly_chart(fig, use_container_width=True,config={"displayModeBar": False}, key=f"env_chart_{col_idx}")

########################## FOOTER #########################
st.markdown("""
<div style="text-align:center;margin-top:2rem;padding-top:1rem;border-top:1px solid #1a1208">
  <span style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;color:#6b5538">
    DATA : CIQUAL · INCA2 · AGRIBALYSE V3
  </span>
</div>
""", unsafe_allow_html=True)
