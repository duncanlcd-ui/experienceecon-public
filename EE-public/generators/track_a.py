# generators/track_a.py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

# ---- Stages & segments ----
STAGES = [
    "Discovery", "Consult_Booking", "Consult_Followup",
    "Treatment", "Support_QA", "Ongoing_Comms", "Repeat_Booking"
]

SEGMENTS = ["VIP_Spender", "OneAndDone", "Browser", "Loyal_Advocate", "HighExpect_Complainer"]

SEG_DEFAULT_MIX = {
    "VIP_Spender": 0.15, "OneAndDone": 0.40, "Browser": 0.25, "Loyal_Advocate": 0.10, "HighExpect_Complainer": 0.10
}

# Behavior knobs (kept simple & explicit)
BEHAV_BASE = {
    "VIP_Spender":           {"repeat_p":0.60, "drop_after_consult_p":0.05, "upsell_tol":0.7,  "price_sens":0.2},
    "OneAndDone":            {"repeat_p":0.10, "drop_after_consult_p":0.15, "upsell_tol":0.3,  "price_sens":0.7},
    "Browser":               {"repeat_p":0.05, "drop_after_consult_p":0.50, "upsell_tol":0.2,  "price_sens":0.6},
    "Loyal_Advocate":        {"repeat_p":0.55, "drop_after_consult_p":0.08, "upsell_tol":0.6,  "price_sens":0.4},
    "HighExpect_Complainer": {"repeat_p":0.15, "drop_after_consult_p":0.12, "upsell_tol":0.25, "price_sens":0.8},
}

# AOV bands (CAD) — defaults; UI may override
SEG_AOV_DEFAULT = {
    "VIP_Spender": (700, 1800),
    "OneAndDone": (300, 700),
    "Browser": (0, 200),
    "Loyal_Advocate": (400, 800),
    "HighExpect_Complainer": (300, 600),
}

# Segment sentiment tilt (-0.5 .. +0.5) added to baseline CSAT
SEG_SENTIMENT_DEFAULT = {
    "VIP_Spender":  +0.10,
    "OneAndDone":   -0.05,
    "Browser":       0.00,
    "Loyal_Advocate":+0.15,
    "HighExpect_Complainer": -0.25,
}

LEX = {
    "vip": ["concierge", "personalized plan", "priority booking"],
    "price": ["transparent pricing", "unexpected add-on", "promo applied"],
    "upsell": ["suggested package", "recommended add-on", "pressure to upgrade"],
    "outcome": ["great result", "visible improvement", "minor improvement", "no change"],
    "staff": ["RN", "aesthetician", "MD", "front desk"],
    "support": ["quick response", "had to follow up", "resolved in one go"],
    "comms": ["helpful offer", "generic newsletter", "timely reminder"],
}

@dataclass
class TrackAParams:
    n_customers: int = 1000
    segment_mix: Dict[str, float] = field(default_factory=lambda: SEG_DEFAULT_MIX.copy())
    aov_overrides: Dict[str, tuple] = field(default_factory=dict)     # {"VIP_Spender": (700,1800)}
    sentiment_bias: Dict[str, float] = field(default_factory=lambda: SEG_SENTIMENT_DEFAULT.copy())
    # Some older UIs may not pass these; our UI filters args anyway
    median_wait_min: float = 10.0
    wait_sigma: float = 0.5
    csat_base: float = 4.2
    seed: int = 42

# ---- helpers ----
def _lognormal_params_from_median(median: float, sigma: float):
    mu = float(np.log(max(median, 0.1)))
    return mu, sigma

def _sample_wait(rng: np.random.Generator, mu: float, sigma: float) -> float:
    return float(rng.lognormal(mean=mu, sigma=sigma))

def _sample_aov(rng: np.random.Generator, seg: str, aov_bands: Dict[str, tuple]) -> float:
    lo, hi = aov_bands.get(seg, SEG_AOV_DEFAULT[seg])
    return float(rng.uniform(lo, hi))

def _segment_for(rng: np.random.Generator, mix: Dict[str, float]) -> str:
    keys = list(mix.keys())
    probs = np.array(list(mix.values()), dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(keys))/len(keys)
    return str(rng.choice(keys, p=probs))

def _text_for(stage: str, seg: str, rng: np.random.Generator,
              price_surprise: float, upsell_pressure: float, outcome_good: float) -> str:
    if stage == "Discovery":
        return f"Heard via friend; expects {rng.choice(['quality care','good deals','natural results'])}"
    if stage == "Consult_Booking":
        return f"Booked consult; {rng.choice(LEX['vip'] if seg=='VIP_Spender' else ['online form','phone'])}"
    if stage == "Consult_Followup":
        phr = rng.choice(LEX["price"]) if price_surprise > 0 else "clear estimate"
        return f"Consult follow-up; {phr}; {rng.choice(['next steps explained','some confusion'])}"
    if stage == "Treatment":
        ups = rng.choice(LEX["upsell"])
        out = rng.choice(LEX["outcome"])
        return f"Treatment by {rng.choice(LEX['staff'])}; {ups}; {out}"
    if stage == "Support_QA":
        return f"Support: {rng.choice(LEX['support'])}"
    if stage == "Ongoing_Comms":
        return f"Comms: {rng.choice(LEX['comms'])}"
    if stage == "Repeat_Booking":
        return f"Booked another service; {rng.choice(['bundle','new treatment','maintenance'])}"
    return "—"

def _csat_adjusted(rng: np.random.Generator, base: float, wait_min: float, seg: str,
                   price_surprise: float, upsell_pressure: float, outcome_good: float,
                   bias: Dict[str, float], behav: Dict[str, Dict[str, float]]) -> float:
    b = behav[seg]
    csat = base + bias.get(seg, 0.0)
    csat -= min(wait_min/30.0, 1.5) * 0.4
    csat -= price_surprise * (0.5 * b["price_sens"])
    csat -= max(0.0, upsell_pressure - b["upsell_tol"]) * 0.4
    csat += outcome_good * 0.6
    return float(np.clip(rng.normal(csat, 0.35), 1.0, 5.0))

# ---- main generator ----
def generate(params: TrackAParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed)
    mu, sigma = _lognormal_params_from_median(params.median_wait_min, params.wait_sigma)
    base = pd.Timestamp.utcnow().floor("min")

    # effective AOV bands (apply overrides)
    aov_bands = SEG_AOV_DEFAULT.copy()
    aov_bands.update(params.aov_overrides)
    behav = BEHAV_BASE

    rows: List[dict] = []

    for i in range(params.n_customers):
        seg = _segment_for(rng, params.segment_mix)
        cust = f"c{i:06d}"
        t = base - pd.Timedelta(minutes=int(rng.integers(0, 21*24*60)))  # last ~3 weeks

        # latent “context” for this customer
        price_surprise = 1.0 if seg in ["OneAndDone","HighExpect_Complainer"] and rng.random() < 0.25 else 0.0
        ups_pressure  = rng.uniform(0.1, 0.8)
        outcome_good  = 1.0 if rng.random() < {"VIP_Spender":0.8,"Loyal_Advocate":0.75}.get(seg, 0.6) else 0.0

        def add(stage: str, spend: float = 0.0):
            nonlocal t
            # sample a fresh wait for THIS stage (so we never rely on outer variables)
            wait_now = _sample_wait(rng, mu, sigma)
            csat_now = _csat_adjusted(
                rng, params.csat_base, wait_now, seg, price_surprise, ups_pressure, outcome_good,
                params.sentiment_bias, behav
            )
            interaction = f"enc{i:06d}-{stage}"
            rows.append({
                "customer_id": cust,
                "interaction_id": interaction,
                "segment": seg,
                "stage": stage,
                "channel": rng.choice(["referral","web","phone","in_person"], p=[0.4,0.35,0.15,0.10]),
                "occurred_at": t,
                "csat": csat_now,
                "wait_minutes": wait_now,
                "spend_cad": spend,
                "text": _text_for(stage, seg, rng, price_surprise, ups_pressure, outcome_good),
            })
            t = t + pd.Timedelta(minutes=max(5.0, wait_now))

        # Stage flow with drop-off & repeat
        add("Discovery")
        add("Consult_Booking")
        add("Consult_Followup")

        if rng.random() < behav[seg]["drop_after_consult_p"]:
            if rng.random() < 0.5:
                add("Ongoing_Comms")
            continue

        spend1 = _sample_aov(rng, seg, aov_bands)
        add("Treatment", spend=spend1)

        if rng.random() < 0.3:
            add("Support_QA")

        add("Ongoing_Comms")

        if rng.random() < behav[seg]["repeat_p"]:
            spend2 = _sample_aov(rng, seg, aov_bands) * rng.uniform(0.6, 1.2)
            add("Repeat_Booking", spend=spend2)

    return pd.DataFrame(rows)
