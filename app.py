import json
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Configuration
# ----------------------------

# FX rates are expressed as: foreign currency units per 1 AUD.
# Example: USD=0.64 means 1 AUD = 0.64 USD, so 1 USD = 1/0.64 AUD.
BASELINE_FX = {"AUD": 1.0, "USD": 0.64, "EUR": 0.53}

DEFAULT_GST_RATE = 0.10

# Sensible default elasticities (RRP-based constant elasticity)
DEFAULT_ELASTICITY_LEVELS = {
    "Low": -0.5,
    "Med": -1.2,
    "High": -2.0,
}

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

DISPLAY_NAME = {
    COL_SUPPLIER: "Supplier",
    COL_CATEGORY: "Product Group",
    COL_SUBCATEGORY: "Product Sub Group",
}

# ----------------------------
# Helpers
# ----------------------------

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == "object":
            df2[c] = df2[c].map(
                lambda v: (
                    "" if pd.isna(v)
                    else (v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v))
                )
            ).astype("string")
    return df2

def _normalise_colname(s: str) -> str:
    s = (s or "").replace("\ufeff", "")
    s = s.strip()
    s = re.sub(r"^\*+", "", s)               # remove leading asterisks
    s = re.sub(r"\s+", " ", s)               # collapse whitespace
    return s.strip()


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
    cols_norm = [_normalise_colname(c).lower() for c in columns]

    def pick(*hints: str) -> Optional[str]:
        for h in hints:
            h = h.lower()
            for i, c in enumerate(cols_norm):
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
        COL_WHOLESALE: pick("default sell price", "wholesale", "sell price"),
        COL_RRP_INC: pick("rrp", "recommended retail", "retail price"),
        COL_QTY: pick("totalsales", "qty", "quantity", "last 12"),
    }


def apply_fx_to_landed_cost(
    landed_cost_aud: pd.Series,
    currency: pd.Series,
    scenario_fx: Dict[str, float],
    baseline_fx: Dict[str, float],
) -> pd.Series:
    """
    Landed cost is assumed to be currently expressed in AUD using BASELINE_FX.
    FX only affects landed cost (per requirement), by scaling for USD/EUR:

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
    Solve for list wholesale price P such that the net wholesale after discount
    hits target wholesaler GM%:

      net = P * (1 - d)
      GM% = (net - cost) / net  =>  net = cost / (1 - GM%)
      => P = cost / (1 - d) / (1 - GM%)
    """
    d = discount_pct.fillna(0.0) / 100.0
    gm = (target_gm_pct / 100.0).clip(lower=-99.0, upper=99.0) / 100.0

    denom = (1.0 - d) * (1.0 - gm)
    denom = denom.replace(0, np.nan)
    return landed_cost / denom


def build_default_rules_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                level="GLOBAL",
                key="ALL",
                rrp_pct_change=0.0,               # enter 10 for 10%
                wholesaler_gm_pp_change=0.0,       # pp change (enter 5 for +5pp)
                wholesaler_gm_target=np.nan,       # target GM% (enter 40 for 40%)
                rrp_ceiling_inc_gst=np.nan,        # AUD inc GST
                elasticity_level="Med",            # Low/Med/High
                elasticity_override=np.nan,        # optional numeric override
                landed_cost_pct_change=0.0,        # enter 10 for 10%
                wholesale_discount_pct=0.0,        # enter 5 for 5%
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
        "elasticity_override",
        "landed_cost_pct_change",
        "wholesale_discount_pct",
    ]
    for col in numeric_cols:
        rules[col] = pd.to_numeric(rules.get(col, np.nan), errors="coerce")

    rules["elasticity_level"] = rules.get("elasticity_level", "Med").astype(str).str.title().str.strip()
    rules.loc[~rules["elasticity_level"].isin(["Low", "Med", "High"]), "elasticity_level"] = "Med"

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
        "elasticity_level",
        "elasticity_override",
        "landed_cost_pct_change",
        "wholesale_discount_pct",
    ]

    global_row = rules[rules["level"] == "GLOBAL"].tail(1).iloc[0]
    out = products.copy()
    for c in param_cols:
        out[c] = global_row.get(c, np.nan)

    # Precedence: Supplier -> Category -> Subcategory -> SKU
    out = _apply_level_overrides(out, rules, "SUPPLIER", COL_SUPPLIER, param_cols)
    out = _apply_level_overrides(out, rules, "CATEGORY", COL_CATEGORY, param_cols)
    out = _apply_level_overrides(out, rules, "SUBCATEGORY", COL_SUBCATEGORY, param_cols)
    out = _apply_level_overrides(out, rules, "SKU", COL_PRODUCT_CODE, param_cols)
    return out


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


def compute_scenario(
    base: pd.DataFrame,
    rules: pd.DataFrame,
    gst_rate: float,
    scenario_fx: Dict[str, float],
    elasticity_levels: Dict[str, float],
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
    df["wholesaler_gm$_old_unit"] = df["wholesale_net_aud"] - df[COL_LANDED]
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
    df["wholesaler_gm%_target_net_pct"] = gm_target

    # Compute new list wholesale to hit target GM% on net wholesale
    df["wholesale_price_new_aud_raw"] = compute_target_wholesale_from_gm(
        df["landed_cost_new_aud"], df["wholesale_discount_pct_eff"], df["wholesaler_gm%_target_net_pct"]
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
    df["wholesaler_gm$_new_unit"] = df["wholesale_net_new_aud"] - df["landed_cost_new_aud"]
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

    # Elasticity selection (level or override)
    e_from_level = df["elasticity_level"].map(elasticity_levels).astype(float)
    e_override = pd.to_numeric(df["elasticity_override"], errors="coerce")
    e = np.where(pd.isna(e_override), e_from_level, e_override)
    df["elasticity"] = pd.Series(e, index=df.index).fillna(elasticity_levels["Med"]).astype(float)

    # Demand modelling: elasticity on RRP (inc GST)
    rrp_ratio = np.where(df[COL_RRP_INC] > 0, df["rrp_new_inc_gst"] / df[COL_RRP_INC], 1.0)
    df["qty_new"] = df[COL_QTY].astype(float) * (rrp_ratio ** df["elasticity"])

    # Financials (net wholesale used)
    df["revenue_old_aud"] = df["wholesale_net_aud"] * df[COL_QTY].astype(float)
    df["gp_old_aud"] = (df["wholesale_net_aud"] - df[COL_LANDED]) * df[COL_QTY].astype(float)

    df["revenue_new_aud"] = df["wholesale_net_new_aud"] * df["qty_new"]
    df["gp_new_aud"] = (df["wholesale_net_new_aud"] - df["landed_cost_new_aud"]) * df["qty_new"]

    # Changes for alerts and tables
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

    # Old/new GM% as percents for display
    df["wholesaler_gm%_old_pct"] = df["wholesaler_gm%_old"] * 100.0
    df["wholesaler_gm%_new_pct"] = df["wholesaler_gm%_new"] * 100.0
    df["retailer_gm%_old_pct"] = df["retailer_gm%_old"] * 100.0
    df["retailer_gm%_new_pct"] = df["retailer_gm%_new"] * 100.0

    # Unit price deltas
    df["wholesale_delta_aud"] = df["wholesale_price_new_aud"] - df[COL_WHOLESALE]
    df["rrp_delta_aud"] = df["rrp_new_inc_gst"] - df[COL_RRP_INC]

    # Total deltas
    df["revenue_delta_aud"] = df["revenue_new_aud"] - df["revenue_old_aud"]
    df["gp_delta_aud"] = df["gp_new_aud"] - df["gp_old_aud"]

    return df


def summarise(df: pd.DataFrame) -> dict:
    tot_rev_old = df["revenue_old_aud"].sum(skipna=True)
    tot_gp_old = df["gp_old_aud"].sum(skipna=True)
    tot_rev_new = df["revenue_new_aud"].sum(skipna=True)
    tot_gp_new = df["gp_new_aud"].sum(skipna=True)

    gm_old = safe_div(tot_gp_old, tot_rev_old)
    gm_new = safe_div(tot_gp_new, tot_rev_new)

    return {
        "rev_old": tot_rev_old,
        "rev_new": tot_rev_new,
        "gp_old": tot_gp_old,
        "gp_new": tot_gp_new,
        "gm_old": gm_old,
        "gm_new": gm_new,
        "units_old": df[COL_QTY].sum(skipna=True),
        "units_new": df["qty_new"].sum(skipna=True),
    }


def group_agg_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, dropna=False).agg(
        sku_count=(COL_PRODUCT_CODE, "count"),
        units_old=(COL_QTY, "sum"),
        units_new=("qty_new", "sum"),
        revenue_old=("revenue_old_aud", "sum"),
        revenue_new=("revenue_new_aud", "sum"),
        gp_old=("gp_old_aud", "sum"),
        gp_new=("gp_new_aud", "sum"),
    ).reset_index()

    g["units_delta"] = g["units_new"] - g["units_old"]
    g["units_delta_%"] = np.where(g["units_old"] > 0, g["units_delta"] / g["units_old"] * 100.0, np.nan)

    g["revenue_delta"] = g["revenue_new"] - g["revenue_old"]
    g["revenue_delta_%"] = np.where(g["revenue_old"] > 0, g["revenue_delta"] / g["revenue_old"] * 100.0, np.nan)

    g["gp_delta"] = g["gp_new"] - g["gp_old"]
    g["gp_delta_%"] = np.where(g["gp_old"] != 0, g["gp_delta"] / g["gp_old"] * 100.0, np.nan)

    g["gm%_old_pct"] = np.where(g["revenue_old"] > 0, g["gp_old"] / g["revenue_old"] * 100.0, np.nan)
    g["gm%_new_pct"] = np.where(g["revenue_new"] > 0, g["gp_new"] / g["revenue_new"] * 100.0, np.nan)
    g["gm%_delta_pp"] = g["gm%_new_pct"] - g["gm%_old_pct"]

    return g


def build_alerts(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    th_w = thresholds.get("wholesale_change_pct", 10.0)
    th_r = thresholds.get("rrp_change_pct", 10.0)
    th_gm = thresholds.get("wholesaler_gm_pp", 5.0)
    th_q = thresholds.get("qty_change_pct", 20.0)

    # New alert: GP$ down even if GM% up
    gp_down_gm_up = (df["gp_new_aud"] < df["gp_old_aud"]) & (df["wholesaler_gm%_new"] > df["wholesaler_gm%_old"])

    flags = [
        ("Negative wholesaler GP$", df["gp_new_aud"] < 0),
        ("GP$ down despite higher GM%", gp_down_gm_up),
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

    return out


def add_alerts_to_df(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    a = build_alerts(df, thresholds)
    merged = df.merge(a[[COL_PRODUCT_CODE, "alerts"]], on=COL_PRODUCT_CODE, how="left", suffixes=("", "_x"))
    merged["alerts"] = merged["alerts"].fillna("")
    return merged


def build_waterfall_gm_by_attribute(df: pd.DataFrame, attr: str, top_n: int = 10) -> go.Figure:
    s = summarise(df)
    rev_old_total = s["rev_old"]
    gm_old_pct = (s["gm_old"] or 0.0) * 100.0
    gm_new_pct = (s["gm_new"] or 0.0) * 100.0

    g = df.groupby(attr, dropna=False).agg(
        gp_old=("gp_old_aud", "sum"),
        gp_new=("gp_new_aud", "sum"),
    ).reset_index()

    g["gp_delta"] = g["gp_new"] - g["gp_old"]
    g["contrib_pp"] = np.where(rev_old_total > 0, g["gp_delta"] / rev_old_total * 100.0, 0.0)

    # Pick top movers (positive and negative), aggregate the rest
    g = g.sort_values("contrib_pp", ascending=False)

    pos = g[g["contrib_pp"] > 0].head(top_n)
    neg = g[g["contrib_pp"] < 0].tail(top_n)  # most negative at the end
    keep = pd.concat([pos, neg], ignore_index=True)

    kept_keys = set(keep[attr].astype(str).tolist())
    other = g[~g[attr].astype(str).isin(kept_keys)]
    other_pp = other["contrib_pp"].sum() if not other.empty else 0.0

    bars = []
    for _, r in keep.sort_values("contrib_pp", ascending=False).iterrows():
        name = str(r[attr]) if str(r[attr]).strip() != "" else "(blank)"
        bars.append((name, float(r["contrib_pp"])))

    if abs(other_pp) > 1e-9:
        bars.append(("Other", float(other_pp)))

    # Denominator effect so the waterfall reconciles to actual GM% new
    interim = gm_old_pct + sum(v for _, v in bars)
    denom_effect = gm_new_pct - interim
    bars.append(("Revenue denominator effect", float(denom_effect)))

    x = ["Original GM%"] + [b[0] for b in bars] + ["New GM%"]
    y = [gm_old_pct] + [b[1] for b in bars] + [0]  # final is computed as total

    measure = ["absolute"] + ["relative"] * len(bars) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            name="GM% waterfall",
            orientation="v",
            measure=measure,
            x=x,
            y=y,
            connector={"line": {"width": 1}},
        )
    )
    fig.update_layout(
        title=f"GM% Waterfall by {DISPLAY_NAME.get(attr, attr)}",
        showlegend=False,
        yaxis_title="GM% (percentage points)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def to_export_df(df: pd.DataFrame, gst_rate: float) -> pd.DataFrame:
    out = df.copy()

    # Helpful export columns: old/change/new pattern
    cols = [
        COL_PRODUCT_CODE,
        COL_PRODUCT_DESC,
        COL_SUPPLIER,
        COL_CATEGORY,
        COL_SUBCATEGORY,
        COL_CURRENCY,
        COL_QTY,
        "qty_new",
        "qty_change_%",
        COL_LANDED,
        "landed_cost_new_aud",
        COL_WHOLESALE,
        "wholesale_price_new_aud",
        "wholesale_delta_aud",
        "wholesale_change_%",
        "wholesale_discount_pct_eff",
        "wholesale_net_aud",
        "wholesale_net_new_aud",
        COL_RRP_INC,
        "rrp_new_inc_gst",
        "rrp_delta_aud",
        "rrp_change_%",
        "rrp_old_ex_gst",
        "rrp_new_ex_gst",
        "wholesaler_gm$_old_unit",
        "wholesaler_gm$_new_unit",
        "wholesaler_gm%_old_pct",
        "wholesaler_gm%_new_pct",
        "wholesaler_gm%_change_pp",
        "retailer_gm%_old_pct",
        "retailer_gm%_new_pct",
        "elasticity_level",
        "elasticity",
        "revenue_old_aud",
        "revenue_new_aud",
        "revenue_delta_aud",
        "gp_old_aud",
        "gp_new_aud",
        "gp_delta_aud",
        "alerts",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()
    out["gst_rate"] = float(gst_rate)
    return out


def build_scenario_payload(
    rules_df: pd.DataFrame,
    gst_rate: float,
    scenario_fx: Dict[str, float],
    thresholds: dict,
    elasticity_levels: Dict[str, float],
) -> dict:
    return {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gst_rate": float(gst_rate),
        "scenario_fx": scenario_fx,
        "thresholds": thresholds,
        "elasticity_levels": elasticity_levels,
        "rules": validate_rules_df(rules_df).to_dict(orient="records"),
    }


def parse_scenario_payload(payload: dict) -> Tuple[pd.DataFrame, float, dict, dict, dict]:
    rules = validate_rules_df(pd.DataFrame(payload.get("rules", [])))
    gst = float(payload.get("gst_rate", DEFAULT_GST_RATE))
    fx = payload.get("scenario_fx", dict(BASELINE_FX))
    thresholds = payload.get("thresholds", {})
    elasticity_levels = payload.get("elasticity_levels", dict(DEFAULT_ELASTICITY_LEVELS))
    return rules, gst, fx, thresholds, elasticity_levels


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Pricing Portfolio Scenario Modeller", layout="wide")
st.title("Pricing Portfolio Scenario Modeller")
st.caption("Upload your product CSV, define scenario rules, and export new Wholesale and RRP pricing with margin, GP$ and demand impacts.")

with st.sidebar:
    st.header("1) Upload")
    uploaded = st.file_uploader("Input CSV", type=["csv"])

    st.header("2) Core settings")

    gst_rate = st.number_input(
        "GST rate (used only to convert RRP inc GST to ex GST for retailer margin)",
        min_value=0.0, max_value=0.25, value=DEFAULT_GST_RATE, step=0.01,
        help="Enter as a decimal. For example 0.10 is 10%."
    )

    st.subheader("FX rates (foreign units per 1 AUD)")
    st.caption("These only affect landed cost for USD and EUR items.")
    fx_usd = st.number_input(
        "USD per 1 AUD", min_value=0.01, max_value=5.0, value=float(BASELINE_FX["USD"]), step=0.01,
        help="Example: 0.64 means 1 AUD = 0.64 USD."
    )
    fx_eur = st.number_input(
        "EUR per 1 AUD", min_value=0.01, max_value=5.0, value=float(BASELINE_FX["EUR"]), step=0.01,
        help="Example: 0.53 means 1 AUD = 0.53 EUR."
    )
    scenario_fx = {"AUD": 1.0, "USD": float(fx_usd), "EUR": float(fx_eur)}

    st.subheader("Demand elasticity levels (RRP-based)")
    st.caption("Elasticity is usually negative: price up, demand down.")
    e_low = st.number_input("Low (less sensitive)", value=float(DEFAULT_ELASTICITY_LEVELS["Low"]), step=0.1, help="Example: -0.5")
    e_med = st.number_input("Med (typical)", value=float(DEFAULT_ELASTICITY_LEVELS["Med"]), step=0.1, help="Example: -1.2")
    e_high = st.number_input("High (very sensitive)", value=float(DEFAULT_ELASTICITY_LEVELS["High"]), step=0.1, help="Example: -2.0")
    elasticity_levels = {"Low": float(e_low), "Med": float(e_med), "High": float(e_high)}

    st.header("3) Alerts thresholds")
    st.caption("Enter 10 for 10%. Enter 5 for 5 percentage points.")
    th_w = st.number_input("Wholesale change alert (± %)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    th_r = st.number_input("RRP change alert (± %)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    th_gm = st.number_input("Wholesaler GM% change alert (± pp)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
    th_q = st.number_input("Demand change alert (± %)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)

    thresholds = dict(
        wholesale_change_pct=float(th_w),
        rrp_change_pct=float(th_r),
        wholesaler_gm_pp=float(th_gm),
        qty_change_pct=float(th_q),
    )

    st.header("4) Scenario files")
    scenario_upload = st.file_uploader("Load a saved scenario JSON to compare", type=["json"], key="scenario_json_upload")


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df0 = pd.read_csv(file)
    df0.columns = [_normalise_colname(c) for c in df0.columns]
    return df0


if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

df_raw = load_csv(uploaded)

# Column mapping UI
st.subheader("Input mapping")
default_map = detect_default_mapping(list(df_raw.columns))

with st.expander("Review or adjust column mapping", expanded=False):
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
        m_cat = select(COL_CATEGORY, "Product group column")
        m_subcat = select(COL_SUBCATEGORY, "Product sub group column")
        m_qty = select(COL_QTY, "Qty sold (last 12 months) column")
    with col3:
        m_landed = select(COL_LANDED, "Landed cost (AUD) column")
        m_wh = select(COL_WHOLESALE, "Wholesale price (AUD, ex GST) column")
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
    "GLOBAL provides defaults. More specific levels override GLOBAL in order: Supplier → Category → Subcategory → SKU.\n"
    "Percentage inputs: enter 10 for 10%. Wholesaler GM% target: enter 40 for 40%."
)

if "rules_df" not in st.session_state:
    st.session_state["rules_df"] = build_default_rules_df()

toolbar = st.columns([1, 3, 8])
with toolbar[0]:
    if st.button("Reset rules"):
        st.session_state["rules_df"] = build_default_rules_df()
with toolbar[1]:
    st.write("")
with toolbar[2]:
    st.write("")

rules_df = validate_rules_df(st.session_state["rules_df"])

edited_rules = st.data_editor(
    rules_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "level": st.column_config.SelectboxColumn("level", options=RULE_LEVELS, required=True),
        "key": st.column_config.TextColumn("key", help="GLOBAL uses ALL. For other levels, enter the exact matching value."),
        "rrp_pct_change": st.column_config.NumberColumn("rrp_pct_change", help="Enter 10 for 10% change to RRP (inc GST)."),
        "wholesaler_gm_pp_change": st.column_config.NumberColumn("wholesaler_gm_pp_change", help="Change in wholesaler GM% in percentage points (enter 5 for +5pp)."),
        "wholesaler_gm_target": st.column_config.NumberColumn("wholesaler_gm_target", help="Target wholesaler GM% (net). Enter 40 for 40%. If set, overrides pp change."),
        "rrp_ceiling_inc_gst": st.column_config.NumberColumn("rrp_ceiling_inc_gst", help="Ceiling on RRP inc GST (AUD)."),
        "elasticity_level": st.column_config.SelectboxColumn("elasticity_level", options=["Low", "Med", "High"], help="Low/Med/High mapped from sidebar values."),
        "elasticity_override": st.column_config.NumberColumn("elasticity_override", help="Optional numeric override (eg -1.2). Overrides the level if set."),
        "landed_cost_pct_change": st.column_config.NumberColumn("landed_cost_pct_change", help="Enter 10 for 10% change to landed cost after FX."),
        "wholesale_discount_pct": st.column_config.NumberColumn("wholesale_discount_pct", help="Average discount off wholesale (enter 5 for 5%). Used for net margin and revenue."),
    },
)

st.session_state["rules_df"] = validate_rules_df(edited_rules)

# Run scenario
scenario_df = compute_scenario(
    df_base,
    st.session_state["rules_df"],
    gst_rate=gst_rate,
    scenario_fx=scenario_fx,
    elasticity_levels=elasticity_levels,
    rounding=True,
)

scenario_df = add_alerts_to_df(scenario_df, thresholds)

# High level metrics
st.subheader("High-level summary")
s = summarise(scenario_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Revenue (AUD)", f"${s['rev_new']:,.0f}", delta=f"${(s['rev_new'] - s['rev_old']):,.0f}")
m2.metric("Gross Profit (AUD)", f"${s['gp_new']:,.0f}", delta=f"${(s['gp_new'] - s['gp_old']):,.0f}")
m3.metric("GM% (revenue-weighted)", f"{(s['gm_new'] or 0.0)*100:,.2f}%", delta=f"{((s['gm_new'] or 0.0)-(s['gm_old'] or 0.0))*100:,.2f} pp")
m4.metric("Units (modelled)", f"{s['units_new']:,.0f}", delta=f"{(s['units_new'] - s['units_old']):,.0f}")

# Waterfall
st.subheader("GM% Waterfall (attribute contribution)")
wcol1, wcol2 = st.columns([2, 1])
with wcol1:
    waterfall_attr = st.selectbox("Waterfall attribute", options=[COL_SUPPLIER, COL_CATEGORY], format_func=lambda x: DISPLAY_NAME.get(x, x))
with wcol2:
    top_n = st.slider("Show top movers (each side)", min_value=3, max_value=25, value=10, step=1)

fig = build_waterfall_gm_by_attribute(scenario_df, attr=waterfall_attr, top_n=top_n)
st.plotly_chart(fig, use_container_width=True)

# Aggregated table (excluding SKUs)
st.subheader("Aggregated results (excluding SKUs)")
g_attr = st.selectbox("Aggregate by", options=[COL_SUPPLIER, COL_CATEGORY, COL_SUBCATEGORY], format_func=lambda x: DISPLAY_NAME.get(x, x))
agg = group_agg_table(scenario_df, by=g_attr)

st.dataframe(
    agg.sort_values("gp_delta", ascending=False),
    use_container_width=True,
    column_config={
        "sku_count": st.column_config.NumberColumn("SKUs", format="%,.0f"),
        "units_old": st.column_config.NumberColumn("Units old", format="%,.0f"),
        "units_new": st.column_config.NumberColumn("Units new", format="%,.0f"),
        "units_delta": st.column_config.NumberColumn("Units change", format="%,.0f"),
        "units_delta_%": st.column_config.NumberColumn("Units change %", format="%.2f%%"),
        "revenue_old": st.column_config.NumberColumn("Revenue old", format="$%,.0f"),
        "revenue_new": st.column_config.NumberColumn("Revenue new", format="$%,.0f"),
        "revenue_delta": st.column_config.NumberColumn("Revenue change", format="$%,.0f"),
        "revenue_delta_%": st.column_config.NumberColumn("Revenue change %", format="%.2f%%"),
        "gp_old": st.column_config.NumberColumn("GP old", format="$%,.0f"),
        "gp_new": st.column_config.NumberColumn("GP new", format="$%,.0f"),
        "gp_delta": st.column_config.NumberColumn("GP change", format="$%,.0f"),
        "gp_delta_%": st.column_config.NumberColumn("GP change %", format="%.2f%%"),
        "gm%_old_pct": st.column_config.NumberColumn("GM% old", format="%.2f%%"),
        "gm%_new_pct": st.column_config.NumberColumn("GM% new", format="%.2f%%"),
        "gm%_delta_pp": st.column_config.NumberColumn("GM% change (pp)", format="%.2f"),
    },
    height=420,
)

# Alerts table
st.subheader("Alerts")
alerts_only = scenario_df[scenario_df["alerts"] != ""].copy()
st.caption(f"{len(alerts_only):,} SKUs flagged under current thresholds.")
st.dataframe(
    alerts_only[[COL_PRODUCT_CODE, COL_PRODUCT_DESC, COL_SUPPLIER, COL_CATEGORY, COL_SUBCATEGORY, "alerts",
                 "gp_old_aud", "gp_new_aud", "gp_delta_aud",
                 "wholesaler_gm%_old_pct", "wholesaler_gm%_new_pct", "wholesaler_gm%_change_pp",
                 "wholesale_change_%", "rrp_change_%", "qty_change_%"
                 ]],
    use_container_width=True,
    column_config={
        "gp_old_aud": st.column_config.NumberColumn("GP old", format="$%,.0f"),
        "gp_new_aud": st.column_config.NumberColumn("GP new", format="$%,.0f"),
        "gp_delta_aud": st.column_config.NumberColumn("GP change", format="$%,.0f"),
        "wholesaler_gm%_old_pct": st.column_config.NumberColumn("Wholesaler GM% old", format="%.2f%%"),
        "wholesaler_gm%_new_pct": st.column_config.NumberColumn("Wholesaler GM% new", format="%.2f%%"),
        "wholesaler_gm%_change_pp": st.column_config.NumberColumn("Wholesaler GM% change (pp)", format="%.2f"),
        "wholesale_change_%": st.column_config.NumberColumn("Wholesale change %", format="%.2f%%"),
        "rrp_change_%": st.column_config.NumberColumn("RRP change %", format="%.2f%%"),
        "qty_change_%": st.column_config.NumberColumn("Demand change %", format="%.2f%%"),
    },
    height=320,
)

# SKU-level table with filters
st.subheader("SKU-level detail (filterable)")

f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
with f1:
    sel_sup = st.multiselect("Filter Supplier", options=sorted([x for x in scenario_df[COL_SUPPLIER].dropna().unique() if str(x).strip() != ""]))
with f2:
    sel_cat = st.multiselect("Filter Product Group", options=sorted([x for x in scenario_df[COL_CATEGORY].dropna().unique() if str(x).strip() != ""]))
with f3:
    sel_sub = st.multiselect("Filter Product Sub Group", options=sorted([x for x in scenario_df[COL_SUBCATEGORY].dropna().unique() if str(x).strip() != ""]))
with f4:
    only_alerts = st.checkbox("Show only alerts", value=False)

df_view = scenario_df.copy()
if sel_sup:
    df_view = df_view[df_view[COL_SUPPLIER].isin(sel_sup)]
if sel_cat:
    df_view = df_view[df_view[COL_CATEGORY].isin(sel_cat)]
if sel_sub:
    df_view = df_view[df_view[COL_SUBCATEGORY].isin(sel_sub)]
if only_alerts:
    df_view = df_view[df_view["alerts"] != ""]

# Slim table by default (reduces payload a lot)
sku_cols_slim = [
    COL_PRODUCT_CODE,
    COL_PRODUCT_DESC,
    COL_SUPPLIER,
    COL_CATEGORY,
    COL_SUBCATEGORY,
    "alerts",
    COL_QTY,
    "qty_new",
    "qty_change_%",
    COL_WHOLESALE,
    "wholesale_price_new_aud",
    COL_RRP_INC,
    "rrp_new_inc_gst",
    "wholesaler_gm%_old_pct",
    "wholesaler_gm%_new_pct",
    "gp_old_aud",
    "gp_new_aud",
    "gp_delta_aud",
]
sku_cols_slim = [c for c in sku_cols_slim if c in df_view.columns]

st.caption("Table is intentionally slim to avoid Streamlit payload limits. Use the SKU drill-down below for full detail.")
st.dataframe(
    make_arrow_safe(df_view[sku_cols_slim]),
    use_container_width=True,
    height=520,
    column_config={
        COL_QTY: st.column_config.NumberColumn("Units old", format="%,.0f"),
        "qty_new": st.column_config.NumberColumn("Units new", format="%,.0f"),
        "qty_change_%": st.column_config.NumberColumn("Units change %", format="%.2f%%"),
        COL_WHOLESALE: st.column_config.NumberColumn("Wholesale old", format="$%,.0f"),
        "wholesale_price_new_aud": st.column_config.NumberColumn("Wholesale new", format="$%,.0f"),
        COL_RRP_INC: st.column_config.NumberColumn("RRP old (inc GST)", format="$%,.0f"),
        "rrp_new_inc_gst": st.column_config.NumberColumn("RRP new (inc GST)", format="$%,.0f"),
        "wholesaler_gm%_old_pct": st.column_config.NumberColumn("Wholesaler GM% old", format="%.2f%%"),
        "wholesaler_gm%_new_pct": st.column_config.NumberColumn("Wholesaler GM% new", format="%.2f%%"),
        "gp_old_aud": st.column_config.NumberColumn("GP old", format="$%,.0f"),
        "gp_new_aud": st.column_config.NumberColumn("GP new", format="$%,.0f"),
        "gp_delta_aud": st.column_config.NumberColumn("GP change", format="$%,.0f"),
    },
)

# Drill-down: select a SKU code and show full detail (only 1 row sent)
st.subheader("Selected SKU drill-down")
sku_pick = st.selectbox(
    "Select a SKU to view full detail",
    options=df_view[COL_PRODUCT_CODE].astype(str).tolist()
)

row = df_view[df_view[COL_PRODUCT_CODE].astype(str) == str(sku_pick)].head(1)
if not row.empty:
    vals = row.iloc[0].to_dict()
    kv = pd.DataFrame(
        {
            "Field": list(vals.keys()),
            "Value": [
                "" if pd.isna(v) else (v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v))
                for v in vals.values()
            ],
        }
    )
    st.dataframe(kv, use_container_width=True, hide_index=True)



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

scenario_payload = build_scenario_payload(
    st.session_state["rules_df"],
    gst_rate=gst_rate,
    scenario_fx=scenario_fx,
    thresholds=thresholds,
    elasticity_levels=elasticity_levels,
)
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
        rules_b, gst_b, fx_b, _, elasticity_b = parse_scenario_payload(payload)

        df_b = compute_scenario(
            df_base,
            rules_b,
            gst_rate=gst_b,
            scenario_fx=fx_b,
            elasticity_levels=elasticity_b,
            rounding=True,
        )
        df_b = add_alerts_to_df(df_b, thresholds)

        sa = summarise(scenario_df)
        sb = summarise(df_b)

        comp = pd.DataFrame([
            {"metric": "Revenue (AUD)", "A": sa["rev_new"], "B": sb["rev_new"], "A-B": sa["rev_new"] - sb["rev_new"]},
            {"metric": "Gross Profit (AUD)", "A": sa["gp_new"], "B": sb["gp_new"], "A-B": sa["gp_new"] - sb["gp_new"]},
            {"metric": "GM% (revenue-weighted)", "A": (sa["gm_new"] or 0)*100, "B": (sb["gm_new"] or 0)*100, "A-B (pp)": ((sa["gm_new"] or 0) - (sb["gm_new"] or 0))*100},
            {"metric": "Units (modelled)", "A": sa["units_new"], "B": sb["units_new"], "A-B": sa["units_new"] - sb["units_new"]},
        ])

        st.subheader("Scenario comparison (current vs loaded JSON)")
        st.dataframe(
            comp,
            use_container_width=True,
            column_config={
                "A": st.column_config.NumberColumn("A", format="$%,.0f"),
                "B": st.column_config.NumberColumn("B", format="$%,.0f"),
                "A-B": st.column_config.NumberColumn("A-B", format="$%,.0f"),
                "A-B (pp)": st.column_config.NumberColumn("A-B (pp)", format="%.2f"),
            },
        )

    except Exception as e:
        st.warning(f"Could not compare scenarios: {e}")
