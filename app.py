import json
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Configuration
# ----------------------------

# FX rates are expressed as: foreign currency units per 1 AUD.
# Example: USD=0.64 means 1 AUD = 0.64 USD, so 1 USD = 1/0.64 AUD.
BASELINE_FX = {"AUD": 1.0, "USD": 0.64, "EUR": 0.53}

DEFAULT_GST_RATE = 0.10

RULE_LEVELS = ["GLOBAL", "SUPPLIER", "CATEGORY", "SUBCATEGORY", "SKU"]

# Column names in the working dataframe (after mapping)
COL_PRODUCT_CODE = "product_code"
COL_PRODUCT_DESC = "product_description"
COL_SUPPLIER = "supplier_code"
COL_CATEGORY = "product_group"
COL_SUBCATEGORY = "product_sub_group"
COL_CURRENCY = "purchase_currency"
COL_LANDED = "landed_cost_aud"          # per unit, AUD
COL_WHOLESALE = "wholesale_price_aud"   # per unit, AUD (ex GST)
COL_RRP_INC = "rrp_inc_gst_aud"         # per unit, AUD (inc GST)
COL_QTY = "qty_12m"                     # units sold over last 12 months


# ----------------------------
# Helpers
# ----------------------------

def _normalise_colname(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\ufeff", "")
    return s


def parse_money(x) -> float:
    """Parse common money strings: "$1,234.50", "1,234", "(123.45)" etc."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    s = (
        s.replace("$", "")
         .replace(",", "")
         .replace("AUD", "")
         .replace("USD", "")
         .replace("EUR", "")
         .strip()
    )
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return np.nan


def nearest_dollar(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(np.round(x, 0))


def safe_div(a: float, b: float) -> float:
    if b == 0 or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a / b


def detect_default_mapping(columns: list[str]) -> Dict[str, Optional[str]]:
    """Best-effort column mapping based on name hints."""
    cols = [c.lower() for c in columns]

    def pick(*hints: str) -> Optional[str]:
        for h in hints:
            for i, c in enumerate(cols):
                if h in c:
                    return columns[i]
        return None

    return {
        COL_PRODUCT_CODE: pick("product code", "sku", "item code", "code"),
        COL_PRODUCT_DESC: pick("product description", "description", "name"),
        COL_SUPPLIER: pick("supplier code", "supplier"),
        COL_CATEGORY: pick("product group", "category", "group"),
        COL_SUBCATEGORY: pick("product sub group", "sub category", "subcategory", "sub group"),
        COL_CURRENCY: pick("currency", "purchase currency"),
        COL_LANDED: pick("landed cost", "landed"),
        COL_WHOLESALE: pick("wholesale", "default sell price", "sell price"),
        COL_RRP_INC: pick("rrp", "recommended retail", "retail price"),
        COL_QTY: pick("qty", "quantity", "12 months", "last 12"),
    }


def apply_fx_to_landed_cost(
    landed_cost_aud: pd.Series,
    currency: pd.Series,
    scenario_fx: Dict[str, float],
    baseline_fx: Dict[str, float],
) -> pd.Series:
    """
    Landed cost is assumed to be currently expressed in AUD using BASELINE_FX.
    FX only affects landed cost (per your requirement), by scaling for USD/EUR:

      new_landed = landed * (baseline_rate / scenario_rate)

    where rates are 'foreign units per 1 AUD'.
    """
    curr = currency.fillna("AUD").astype(str).str.upper()
    mult = pd.Series(1.0, index=landed_cost_aud.index, dtype="float64")

    for ccy in ["USD", "EUR"]:
        base = float(baseline_fx.get(ccy, 1.0))
        scen = float(scenario_fx.get(ccy, base))
        if scen <= 0:
            scen = base
        m = base / scen
        mult = np.where(curr == ccy, m, mult)

    return landed_cost_aud.astype(float) * mult


def compute_target_wholesale_from_gm(
    landed_cost: pd.Series,
    discount_pct: pd.Series,
    target_gm_pct: pd.Series,
) -> pd.Series:
    """
    Solve for list wholesale price P such that the *net* wholesale after discount
    hits target wholesaler GM%:

      net = P * (1 - d)
      GM% = (net - cost) / net  =>  net = cost / (1 - GM%)
      => P = cost / (1 - d) / (1 - GM%)
    """
    d = discount_pct.fillna(0.0) / 100.0
    gm = (target_gm_pct / 100.0).clip(lower=-0.99, upper=0.99)

    denom = (1.0 - d) * (1.0 - gm)
    denom = denom.replace(0, np.nan)
    return landed_cost / denom


def build_default_rules_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                level="GLOBAL",
                key="ALL",
                rrp_pct_change=0.0,
                wholesaler_gm_pp_change=0.0,
                wholesaler_gm_target=np.nan,
                rrp_ceiling_inc_gst=np.nan,
                elasticity=-1.2,
                landed_cost_pct_change=0.0,
                wholesale_discount_pct=0.0,
            )
        ]
    )


def validate_rules_df(rules: pd.DataFrame) -> pd.DataFrame:
    rules = rules.copy()
    if rules.empty:
        rules = build_default_rules_df()

    rules["level"] = rules.get("level", "GLOBAL").astype(str).str.upper().str.strip()
    rules["key"] = rules.get("key", "").astype(str).str.strip()

    numeric_cols = [
        "rrp_pct_change",
        "wholesaler_gm_pp_change",
        "wholesaler_gm_target",
        "rrp_ceiling_inc_gst",
        "elasticity",
        "landed_cost_pct_change",
        "wholesale_discount_pct",
    ]
    for col in numeric_cols:
        rules[col] = pd.to_numeric(rules.get(col, np.nan), errors="coerce")

    if not (rules["level"] == "GLOBAL").any():
        rules = pd.concat([build_default_rules_df(), rules], ignore_index=True)

    return rules


def _apply_level_overrides(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    level: str,
    df_key_col: Optional[str],
    param_cols: list[str],
) -> pd.DataFrame:
    level_rules = rules[rules["level"] == level].copy()
    if level_rules.empty or df_key_col is None:
        return df

    level_rules = (
        level_rules.dropna(subset=["key"])
        .sort_index()
        .drop_duplicates(subset=["key"], keep="last")
    )

    merge_cols = ["key"] + param_cols
    m = df.merge(
        level_rules[merge_cols],
        how="left",
        left_on=df_key_col,
        right_on="key",
        suffixes=("", "_rule"),
    )

    for c in param_cols:
        rc = f"{c}_rule"
        if rc in m.columns:
            mask = m[rc].notna()
            m.loc[mask, c] = m.loc[mask, rc]
            m = m.drop(columns=[rc])

    m = m.drop(columns=["key"])
    return m


def apply_rules_with_precedence(products: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rules = validate_rules_df(rules)

    param_cols = [
        "rrp_pct_change",
        "wholesaler_gm_pp_change",
        "wholesaler_gm_target",
        "rrp_ceiling_inc_gst",
        "elasticity",
        "landed_cost_pct_change",
        "wholesale_discount_pct",
    ]

    global_row = rules[rules["level"] == "GLOBAL"].tail(1).iloc[0]
    out = products.copy()
    for c in param_cols:
        out[c] = global_row.get(c, np.nan)

    out = _apply_level_overrides(out, rules, "SUPPLIER", COL_SUPPLIER, param_cols)
    out = _apply_level_overrides(out, rules, "CATEGORY", COL_CATEGORY, param_cols)
    out = _apply_level_overrides(out, rules, "SUBCATEGORY", COL_SUBCATEGORY, param_cols)
    out = _apply_level_overrides(out, rules, "SKU", COL_PRODUCT_CODE, param_cols)
    return out


def compute_scenario(
    base: pd.DataFrame,
    rules: pd.DataFrame,
    gst_rate: float,
    scenario_fx: Dict[str, float],
    baseline_fx: Dict[str, float] = BASELINE_FX,
    rounding: bool = True,
) -> pd.DataFrame:
    df = apply_rules_with_precedence(base, rules)

    # FX impact on landed cost only
    df["landed_cost_fx_aud"] = apply_fx_to_landed_cost(df[COL_LANDED], df[COL_CURRENCY], scenario_fx, baseline_fx)

    # Landed cost adjustment (post-FX)
    df["landed_cost_new_aud"] = df["landed_cost_fx_aud"] * (1.0 + (df["landed_cost_pct_change"].fillna(0.0) / 100.0))

    # Discount on wholesale (used for net wholesale)
    df["wholesale_discount_pct_eff"] = df["wholesale_discount_pct"].fillna(0.0).clip(lower=0.0, upper=99.0)
    df["wholesale_net_aud"] = df[COL_WHOLESALE] * (1.0 - df["wholesale_discount_pct_eff"] / 100.0)

    # Current wholesaler margin (net)
    df["wholesaler_gm$_old"] = df["wholesale_net_aud"] - df[COL_LANDED]
    df["wholesaler_gm%_old"] = np.where(
        df["wholesale_net_aud"] > 0,
        (df["wholesale_net_aud"] - df[COL_LANDED]) / df["wholesale_net_aud"],
        np.nan,
    )

    # Determine target wholesaler GM% (net)
    gm_old_pct = df["wholesaler_gm%_old"] * 100.0
    gm_target = df["wholesaler_gm_target"].copy()
    gm_target = np.where(
        pd.isna(gm_target),
        gm_old_pct + df["wholesaler_gm_pp_change"].fillna(0.0),
        gm_target,
    )
    gm_target = pd.Series(gm_target, index=df.index, dtype="float64").clip(lower=-99.0, upper=99.0)
    df["wholesaler_gm%_target_net"] = gm_target

    # Compute new list wholesale to hit target GM% on net wholesale
    df["wholesale_price_new_aud_raw"] = compute_target_wholesale_from_gm(
        df["landed_cost_new_aud"], df["wholesale_discount_pct_eff"], df["wholesaler_gm%_target_net"]
    )

    # RRP change and ceiling (RRP includes GST)
    df["rrp_new_inc_gst_raw"] = df[COL_RRP_INC] * (1.0 + (df["rrp_pct_change"].fillna(0.0) / 100.0))

    ceiling = df["rrp_ceiling_inc_gst"]
    mask_ceiling = ceiling.notna()
    df.loc[mask_ceiling, "rrp_new_inc_gst_raw"] = np.minimum(df.loc[mask_ceiling, "rrp_new_inc_gst_raw"], ceiling[mask_ceiling])

    # Rounding
    if rounding:
        df["wholesale_price_new_aud"] = df["wholesale_price_new_aud_raw"].apply(nearest_dollar)
        df["rrp_new_inc_gst"] = df["rrp_new_inc_gst_raw"].apply(nearest_dollar)
    else:
        df["wholesale_price_new_aud"] = df["wholesale_price_new_aud_raw"]
        df["rrp_new_inc_gst"] = df["rrp_new_inc_gst_raw"]

    # Net wholesale (new) after discount
    df["wholesale_net_new_aud"] = df["wholesale_price_new_aud"] * (1.0 - df["wholesale_discount_pct_eff"] / 100.0)

    # New wholesaler margin (net)
    df["wholesaler_gm$_new"] = df["wholesale_net_new_aud"] - df["landed_cost_new_aud"]
    df["wholesaler_gm%_new"] = np.where(
        df["wholesale_net_new_aud"] > 0,
        (df["wholesale_net_new_aud"] - df["landed_cost_new_aud"]) / df["wholesale_net_new_aud"],
        np.nan,
    )

    # Retailer margin (ex GST), using net wholesale
    gst = float(gst_rate)
    df["rrp_old_ex_gst"] = df[COL_RRP_INC] / (1.0 + gst)
    df["rrp_new_ex_gst"] = df["rrp_new_inc_gst"] / (1.0 + gst)

    df["retailer_gm%_old"] = np.where(
        df["rrp_old_ex_gst"] > 0,
        (df["rrp_old_ex_gst"] - df["wholesale_net_aud"]) / df["rrp_old_ex_gst"],
        np.nan,
    )
    df["retailer_gm%_new"] = np.where(
        df["rrp_new_ex_gst"] > 0,
        (df["rrp_new_ex_gst"] - df["wholesale_net_new_aud"]) / df["rrp_new_ex_gst"],
        np.nan,
    )

    # Demand modelling: elasticity on RRP (inc GST)
    e = df["elasticity"].fillna(-1.2).astype(float)
    rrp_ratio = np.where(df[COL_RRP_INC] > 0, df["rrp_new_inc_gst"] / df[COL_RRP_INC], 1.0)
    df["qty_new"] = df[COL_QTY].astype(float) * (rrp_ratio ** e)

    # Financials (net wholesale used)
    df["revenue_old_aud"] = df["wholesale_net_aud"] * df[COL_QTY].astype(float)
    df["gp_old_aud"] = (df["wholesale_net_aud"] - df[COL_LANDED]) * df[COL_QTY].astype(float)

    df["revenue_new_aud"] = df["wholesale_net_new_aud"] * df["qty_new"]
    df["gp_new_aud"] = (df["wholesale_net_new_aud"] - df["landed_cost_new_aud"]) * df["qty_new"]

    # Changes for alerts
    df["wholesale_change_%"] = np.where(
        df[COL_WHOLESALE] > 0,
        (df["wholesale_price_new_aud"] - df[COL_WHOLESALE]) / df[COL_WHOLESALE] * 100.0,
        np.nan,
    )
    df["rrp_change_%"] = np.where(
        df[COL_RRP_INC] > 0,
        (df["rrp_new_inc_gst"] - df[COL_RRP_INC]) / df[COL_RRP_INC] * 100.0,
        np.nan,
    )
    df["wholesaler_gm%_change_pp"] = (df["wholesaler_gm%_new"] - df["wholesaler_gm%_old"]) * 100.0
    df["qty_change_%"] = np.where(df[COL_QTY] > 0, (df["qty_new"] - df[COL_QTY]) / df[COL_QTY] * 100.0, np.nan)

    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    tot_rev_old = df["revenue_old_aud"].sum(skipna=True)
    tot_gp_old = df["gp_old_aud"].sum(skipna=True)
    tot_rev_new = df["revenue_new_aud"].sum(skipna=True)
    tot_gp_new = df["gp_new_aud"].sum(skipna=True)

    gm_old = safe_div(tot_gp_old, tot_rev_old)
    gm_new = safe_div(tot_gp_new, tot_rev_new)

    return pd.DataFrame(
        [
            {"metric": "Revenue (AUD)", "old": tot_rev_old, "new": tot_rev_new, "change": tot_rev_new - tot_rev_old},
            {"metric": "Gross Profit (AUD)", "old": tot_gp_old, "new": tot_gp_new, "change": tot_gp_new - tot_gp_old},
            {"metric": "GM% (Revenue-weighted)", "old": gm_old, "new": gm_new, "change": gm_new - gm_old},
            {"metric": "Units (12m modelled)", "old": df[COL_QTY].sum(skipna=True), "new": df["qty_new"].sum(skipna=True), "change": df["qty_new"].sum(skipna=True) - df[COL_QTY].sum(skipna=True)},
        ]
    )


def to_export_df(df: pd.DataFrame, gst_rate: float) -> pd.DataFrame:
    cols = [
        COL_PRODUCT_CODE,
        COL_PRODUCT_DESC,
        COL_SUPPLIER,
        COL_CATEGORY,
        COL_SUBCATEGORY,
        COL_CURRENCY,
        COL_QTY,
        COL_LANDED,
        "landed_cost_new_aud",
        COL_WHOLESALE,
        "wholesale_price_new_aud",
        "wholesale_discount_pct_eff",
        "wholesale_net_aud",
        "wholesale_net_new_aud",
        COL_RRP_INC,
        "rrp_new_inc_gst",
        "rrp_old_ex_gst",
        "rrp_new_ex_gst",
        "wholesaler_gm$_old",
        "wholesaler_gm$_new",
        "wholesaler_gm%_old",
        "wholesaler_gm%_new",
        "retailer_gm%_old",
        "retailer_gm%_new",
        "elasticity",
        "qty_new",
        "revenue_old_aud",
        "revenue_new_aud",
        "gp_old_aud",
        "gp_new_aud",
        "wholesale_change_%",
        "rrp_change_%",
        "wholesaler_gm%_change_pp",
        "qty_change_%",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["gst_rate"] = float(gst_rate)
    return out


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, dropna=False).agg(
        sku_count=(COL_PRODUCT_CODE, "count"),
        units_old=(COL_QTY, "sum"),
        units_new=("qty_new", "sum"),
        revenue_old=("revenue_old_aud", "sum"),
        revenue_new=("revenue_new_aud", "sum"),
        gp_old=("gp_old_aud", "sum"),
        gp_new=("gp_new_aud", "sum"),
    )
    g["gm%_old"] = np.where(g["revenue_old"] > 0, g["gp_old"] / g["revenue_old"], np.nan)
    g["gm%_new"] = np.where(g["revenue_new"] > 0, g["gp_new"] / g["revenue_new"], np.nan)
    g["gm%_change_pp"] = (g["gm%_new"] - g["gm%_old"]) * 100.0
    return g.reset_index()


def build_alerts(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    th_w = thresholds.get("wholesale_change_pct", 10.0)
    th_r = thresholds.get("rrp_change_pct", 10.0)
    th_gm = thresholds.get("wholesaler_gm_pp", 5.0)
    th_q = thresholds.get("qty_change_pct", 20.0)

    flags = [
        ("Negative wholesaler GM$", df["wholesaler_gm$_new"] < 0),
        ("Wholesale change exceeds threshold", df["wholesale_change_%"].abs() >= th_w),
        ("RRP change exceeds threshold", df["rrp_change_%"].abs() >= th_r),
        ("Wholesaler GM% change exceeds threshold", df["wholesaler_gm%_change_pp"].abs() >= th_gm),
        ("Demand change exceeds threshold", df["qty_change_%"].abs() >= th_q),
    ]

    out = df[[COL_PRODUCT_CODE, COL_PRODUCT_DESC, COL_SUPPLIER, COL_CATEGORY, COL_SUBCATEGORY]].copy()
    out["alerts"] = ""

    for name, mask in flags:
        mask = mask.fillna(False)
        out.loc[mask, "alerts"] = out.loc[mask, "alerts"].where(out.loc[mask, "alerts"] == "", out.loc[mask, "alerts"] + " | ") + name

    return out[out["alerts"] != ""]


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Pricing Portfolio Scenario Modeller", layout="wide")

st.title("Pricing Portfolio Scenario Modeller")
st.caption("Upload your product CSV, define scenario rules, and export new Wholesale and RRP pricing with margin and demand impacts.")

with st.sidebar:
    st.header("1) Upload")
    uploaded = st.file_uploader("Input CSV", type=["csv"])

    st.header("2) Core settings")
    gst_rate = st.number_input(
        "GST rate (used to convert RRP inc GST to ex GST for retailer margin)",
        min_value=0.0, max_value=0.25, value=DEFAULT_GST_RATE, step=0.01
    )

    st.subheader("FX rates (foreign units per 1 AUD)")
    fx_usd = st.number_input("USD per 1 AUD", min_value=0.01, max_value=5.0, value=float(BASELINE_FX["USD"]), step=0.01)
    fx_eur = st.number_input("EUR per 1 AUD", min_value=0.01, max_value=5.0, value=float(BASELINE_FX["EUR"]), step=0.01)
    scenario_fx = {"AUD": 1.0, "USD": float(fx_usd), "EUR": float(fx_eur)}

    st.header("3) Alerts")
    th_w = st.number_input("Alert if Wholesale change ≥ (%)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    th_r = st.number_input("Alert if RRP change ≥ (%)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    th_gm = st.number_input("Alert if Wholesaler GM% change ≥ (pp)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
    th_q = st.number_input("Alert if Demand change ≥ (%)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)

    thresholds = dict(
        wholesale_change_pct=float(th_w),
        rrp_change_pct=float(th_r),
        wholesaler_gm_pp=float(th_gm),
        qty_change_pct=float(th_q),
    )

    st.header("4) Scenario files")
    scenario_upload = st.file_uploader("Load a saved scenario (JSON) to compare", type=["json"], key="scenario_json_upload")


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df0 = pd.read_csv(file)
    df0.columns = [_normalise_colname(c) for c in df0.columns]
    return df0


def standardise_input(df_raw: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [_normalise_colname(c) for c in df.columns]

    out = pd.DataFrame()
    for k, src in mapping.items():
        out[k] = df[src] if (src is not None and src in df.columns) else np.nan

    out[COL_LANDED] = out[COL_LANDED].apply(parse_money)
    out[COL_WHOLESALE] = out[COL_WHOLESALE].apply(parse_money)
    out[COL_RRP_INC] = out[COL_RRP_INC].apply(parse_money)
    out[COL_QTY] = pd.to_numeric(out[COL_QTY], errors="coerce")

    for c in [COL_PRODUCT_CODE, COL_PRODUCT_DESC, COL_SUPPLIER, COL_CATEGORY, COL_SUBCATEGORY, COL_CURRENCY]:
        out[c] = out[c].astype(str).replace({"nan": ""}).str.strip()

    out[COL_CURRENCY] = out[COL_CURRENCY].str.upper().replace({"": "AUD"})
    return out


def build_scenario_payload(rules_df: pd.DataFrame) -> dict:
    return {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gst_rate": float(gst_rate),
        "scenario_fx": scenario_fx,
        "thresholds": thresholds,
        "rules": validate_rules_df(rules_df).to_dict(orient="records"),
    }


def parse_scenario_payload(payload: dict) -> tuple[pd.DataFrame, float, dict]:
    rules = validate_rules_df(pd.DataFrame(payload.get("rules", [])))
    gst = float(payload.get("gst_rate", DEFAULT_GST_RATE))
    fx = payload.get("scenario_fx", dict(BASELINE_FX))
    return rules, gst, fx


if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

df_raw = load_csv(uploaded)

# Column mapping UI
st.subheader("Input mapping")
default_map = detect_default_mapping(list(df_raw.columns))

with st.expander("Review / adjust column mapping", expanded=False):
    col1, col2, col3 = st.columns(3)

    def select(col, label, help_txt=None):
        options = ["(not in file)"] + list(df_raw.columns)
        default = default_map.get(col)
        idx = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=idx, help=help_txt)

    with col1:
        m_code = select(COL_PRODUCT_CODE, "Product code column")
        m_desc = select(COL_PRODUCT_DESC, "Product description column")
        m_supplier = select(COL_SUPPLIER, "Supplier code column")
        m_currency = select(COL_CURRENCY, "Purchase currency column")
    with col2:
        m_cat = select(COL_CATEGORY, "Product category column (Product group)")
        m_subcat = select(COL_SUBCATEGORY, "Sub category column (Product sub group)")
        m_qty = select(COL_QTY, "Qty sold (last 12 months) column")
    with col3:
        m_landed = select(COL_LANDED, "Landed cost (AUD) column")
        m_wh = select(COL_WHOLESALE, "Wholesale (AUD, ex GST) column")
        m_rrp = select(COL_RRP_INC, "RRP (AUD, inc GST) column")

mapping = {
    COL_PRODUCT_CODE: None if m_code == "(not in file)" else m_code,
    COL_PRODUCT_DESC: None if m_desc == "(not in file)" else m_desc,
    COL_SUPPLIER: None if m_supplier == "(not in file)" else m_supplier,
    COL_CATEGORY: None if m_cat == "(not in file)" else m_cat,
    COL_SUBCATEGORY: None if m_subcat == "(not in file)" else m_subcat,
    COL_CURRENCY: None if m_currency == "(not in file)" else m_currency,
    COL_LANDED: None if m_landed == "(not in file)" else m_landed,
    COL_WHOLESALE: None if m_wh == "(not in file)" else m_wh,
    COL_RRP_INC: None if m_rrp == "(not in file)" else m_rrp,
    COL_QTY: None if m_qty == "(not in file)" else m_qty,
}

df_base = standardise_input(df_raw, mapping)

missing_core = [k for k in [COL_PRODUCT_CODE, COL_SUPPLIER, COL_CATEGORY, COL_SUBCATEGORY, COL_CURRENCY, COL_LANDED, COL_WHOLESALE, COL_RRP_INC, COL_QTY] if df_base[k].isna().all()]
if missing_core:
    st.error(f"Missing required columns in mapping: {', '.join(missing_core)}")
    st.stop()

# Rules editor
st.subheader("Scenario rules (overrides by precedence)")
st.caption(
    "GLOBAL provides defaults. More specific levels override GLOBAL: Supplier → Category → Subcategory → SKU.\n"
    "Wholesaler GM% is targeted on NET wholesale (after any average discount). New prices are rounded to the nearest dollar."
)

if "rules_df" not in st.session_state:
    st.session_state["rules_df"] = build_default_rules_df()

toolbar = st.columns([1, 1, 6])
with toolbar[0]:
    if st.button("Reset rules"):
        st.session_state["rules_df"] = build_default_rules_df()
with toolbar[1]:
    st.write("")

rules_df = validate_rules_df(st.session_state["rules_df"])

edited_rules = st.data_editor(
    rules_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "level": st.column_config.SelectboxColumn("level", options=RULE_LEVELS, required=True),
        "key": st.column_config.TextColumn("key", help="For GLOBAL use ALL. For other levels, the exact value to match."),
        "rrp_pct_change": st.column_config.NumberColumn("rrp_pct_change", help="% change to RRP (inc GST)"),
        "wholesaler_gm_pp_change": st.column_config.NumberColumn("wholesaler_gm_pp_change", help="Change in wholesaler GM% (pp), net of discount"),
        "wholesaler_gm_target": st.column_config.NumberColumn("wholesaler_gm_target", help="Target wholesaler GM% (net). If set, overrides pp change."),
        "rrp_ceiling_inc_gst": st.column_config.NumberColumn("rrp_ceiling_inc_gst", help="Ceiling on RRP inc GST (AUD)"),
        "elasticity": st.column_config.NumberColumn("elasticity", help="Demand elasticity applied to RRP change (usually negative)"),
        "landed_cost_pct_change": st.column_config.NumberColumn("landed_cost_pct_change", help="% change to landed cost after FX"),
        "wholesale_discount_pct": st.column_config.NumberColumn("wholesale_discount_pct", help="Average discount off wholesale (used for net margin and revenue)"),
    },
)

st.session_state["rules_df"] = validate_rules_df(edited_rules)

# Run scenario
scenario_df = compute_scenario(
    df_base,
    st.session_state["rules_df"],
    gst_rate=gst_rate,
    scenario_fx=scenario_fx,
    rounding=True,
)

# Summary
st.subheader("High-level summary")
st.dataframe(summarise(scenario_df), use_container_width=True)

# Grouping
st.subheader("Breakdown")
group_col = st.selectbox(
    "Group results by",
    options=[COL_SUPPLIER, COL_CATEGORY, COL_SUBCATEGORY],
    format_func=lambda x: {COL_SUPPLIER: "Supplier", COL_CATEGORY: "Product category", COL_SUBCATEGORY: "Sub category"}.get(x, x),
)
st.dataframe(group_summary(scenario_df, by=group_col), use_container_width=True)

# Alerts
st.subheader("Alerts")
alerts_df = build_alerts(scenario_df, thresholds)
st.caption(f"{len(alerts_df):,} SKUs flagged under current thresholds.")
st.dataframe(alerts_df, use_container_width=True, height=320)

# Detailed table
st.subheader("SKU-level detail (preview)")
preview_cols = [
    COL_PRODUCT_CODE, COL_PRODUCT_DESC, COL_SUPPLIER, COL_CATEGORY, COL_SUBCATEGORY, COL_CURRENCY,
    COL_QTY, COL_LANDED, "landed_cost_new_aud",
    COL_WHOLESALE, "wholesale_price_new_aud",
    COL_RRP_INC, "rrp_new_inc_gst",
    "wholesale_change_%", "rrp_change_%",
    "wholesaler_gm%_old", "wholesaler_gm%_new",
    "retailer_gm%_old", "retailer_gm%_new",
    "qty_change_%",
]
preview_cols = [c for c in preview_cols if c in scenario_df.columns]
st.dataframe(scenario_df[preview_cols].head(500), use_container_width=True, height=420)

# Export
st.subheader("Export")
export_df = to_export_df(scenario_df, gst_rate=gst_rate)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
export_name = f"pricing_scenario_export_{ts}.csv"
st.download_button(
    "Download CSV export (all SKUs)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=export_name,
    mime="text/csv",
)

scenario_payload = build_scenario_payload(st.session_state["rules_df"])
scenario_name = f"pricing_scenario_{ts}.json"
st.download_button(
    "Download scenario JSON (rules + settings)",
    data=json.dumps(scenario_payload, indent=2).encode("utf-8"),
    file_name=scenario_name,
    mime="application/json",
)

# Scenario comparison (optional)
if scenario_upload is not None:
    try:
        payload = json.loads(scenario_upload.getvalue().decode("utf-8"))
        rules_b, gst_b, fx_b = parse_scenario_payload(payload)
        df_b = compute_scenario(df_base, rules_b, gst_rate=gst_b, scenario_fx=fx_b, rounding=True)

        sum_a = summarise(scenario_df).rename(columns={"old": "A_old", "new": "A_new", "change": "A_change"})
        sum_b = summarise(df_b).rename(columns={"old": "B_old", "new": "B_new", "change": "B_change"})
        comp = sum_a.merge(sum_b, on="metric", how="outer")

        st.subheader("Scenario comparison (current vs loaded JSON)")
        st.dataframe(comp, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not compare scenarios: {e}")
