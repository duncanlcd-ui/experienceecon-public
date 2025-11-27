# pipelines/text_nlp.py
# Simple NLP pipeline: VADER sentiment + keyword rules for topics & stage_guess + churn_risk

from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None  # will fall back to neutral if not available


# -----------------------------
# 1) Keyword rules (edit safely)
# -----------------------------
# Medispa topics you listed: Pricing, Wait_Time, Outcome, Upsell, Staff_Behavior
TOPIC_RULES_RAW: Dict[str, List[str]] = {
    "Pricing": [
        r"\bprice(?:s|d|y)?\b",
        r"\bquote(?:d|)\b",
        r"\bcost(?:s|ly)?\b",
        r"\bexpensive\b",
        r"\bover[- ]?charge(?:d|)\b",
        r"\bdeposit\b",
    ],
    "Wait_Time": [
        r"\bwait(?:ed|ing)?\b",
        r"\bdelay(?:ed|)\b",
        r"\brun(?:ning)?\s*late\b",
        r"\bover\s*\d+\s*min\b",
        r"\bqueue\b",
    ],
    "Outcome": [
        r"\bresult(?:s)?\b",
        r"\boutcome(?:s)?\b",
        r"\bsettle(?:d|)\b",
        r"\bbruis(?:e|ing)\b",
        r"\bredness\b",
        r"\bhealing\b",
    ],
    "Upsell": [
        r"\bupsell\b",
        r"\badd[- ]?on(?:s)?\b",
        r"\bpackage(?:s)?\b",
        r"\bmembership\b",
        r"\bpressure(?:d|)\b",
        r"\bpush(?:ing|ed)?\b",
    ],
    "Staff_Behavior": [
        r"\brude\b",
        r"\bunhelpful\b",
        r"\battitude\b",
        r"\bfront\s*desk\b",
        r"\bcoaching\b",
        r"\bscript(?:ing)?\b",
    ],
}

# Lightweight stage hints (your Track A stages)
STAGE_RULES_RAW: Dict[str, List[str]] = {
    "Discovery":         [r"\bdiscovery\b", r"\bresearch\b", r"\bfound\b", r"\bheard via\b", r"\breferral\b"],
    "Consult_Booking":   [r"\bbook(?:ing|)\b", r"\bconsult(?:ation)?\b", r"\brequest\b", r"\bavailability\b", r"\bschedule\b"],
    "Consult_Followup":  [r"\bfollow[- ]?up\b", r"\bclarif(?:y|ication)\b", r"\bquote\b", r"\bestimate\b"],
    "Treatment":         [r"\btreatment\b", r"\bprocedure\b", r"\binject(?:ion|ables?)\b", r"\blaser\b", r"\bfillers?\b"],
    "Support_QA":        [r"\bsupport\b", r"\bquestion\b", r"\bhelp\b", r"\bissue\b", r"\bconcern\b"],
    "Ongoing_Comms":     [r"\bnewsletter\b", r"\boffer\b", r"\breminder\b", r"\bemail\b", r"\btext message\b"],
    "Repeat_Booking":    [r"\brepeat\b", r"\breturn\b", r"\brebook\b", r"\bnext (visit|session)\b"],
}

@dataclass
class _CompiledRules:
    topics: List[Tuple[str, re.Pattern]]
    stages: List[Tuple[str, re.Pattern]]

def _compile_rules() -> _CompiledRules:
    def comp(d: Dict[str, List[str]]) -> List[Tuple[str, re.Pattern]]:
        items: List[Tuple[str, re.Pattern]] = []
        for label, pats in d.items():
            for p in pats:
                items.append((label, re.compile(p, flags=re.IGNORECASE)))
        return items
    return _CompiledRules(
        topics=comp(TOPIC_RULES_RAW),
        stages=comp(STAGE_RULES_RAW),
    )

_RULES = _compile_rules()


# -----------------------------
# 2) Helpers
# -----------------------------
def _sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"

def _first_match_label(text: str, compiled: List[Tuple[str, re.Pattern]], default: str) -> str:
    for label, pat in compiled:
        if pat.search(text):
            return label
    return default


# -----------------------------
# 3) Main pipeline function
# -----------------------------
def analyze_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: sentiment_score, sentiment_label, topic, stage_guess, churn_risk
    Leaves existing cols unchanged. Does not drop rows.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Ensure we have a text column
    if "text" not in out.columns:
        out["text"] = ""
    out["text"] = out["text"].fillna("").astype(str)

    # Sentiment (VADER)
    if _VADER is not None:
        scores = out["text"].map(lambda s: _VADER.polarity_scores(s).get("compound", 0.0))
    else:
        scores = pd.Series([0.0] * len(out), index=out.index)
    out["sentiment_score"] = scores
    out["sentiment_label"] = scores.map(_sentiment_label)

    # Topic & stage_guess by keyword rules
    out["topic"] = out["text"].map(lambda t: _first_match_label(t, _RULES.topics, default="other"))
    # Prefer provided stage; otherwise guess
    if "stage" in out.columns:
        out["stage_guess"] = out["stage"].fillna("").astype(str)
        # For blank stage values, try to guess
        mask_blank = out["stage_guess"].eq("") | out["stage_guess"].isna()
        out.loc[mask_blank, "stage_guess"] = out.loc[mask_blank, "text"].map(
            lambda t: _first_match_label(t, _RULES.stages, default="Discovery")
        )
    else:
        out["stage_guess"] = out["text"].map(lambda t: _first_match_label(t, _RULES.stages, default="Discovery"))

    # Churn risk (simple heuristic 0..3)
    # +1 if negative sentiment; +1 if pricing/wait/staff topic; +1 if low CSAT or high wait
    risk = pd.Series(0, index=out.index, dtype=int)
    risk = risk + (out["sentiment_label"].eq("negative")).astype(int)

    risk_topics = {"Pricing", "Wait_Time", "Staff_Behavior"}
    risk = risk + (out["topic"].isin(risk_topics)).astype(int)

    csat = pd.to_numeric(out["csat"], errors="coerce") if "csat" in out.columns else pd.Series(pd.NA, index=out.index)
    wait = pd.to_numeric(out["wait_minutes"], errors="coerce") if "wait_minutes" in out.columns else pd.Series(pd.NA, index=out.index)
    risk = risk + ((csat.notna() & (csat < 3.0)) | (wait.notna() & (wait > 20))).astype(int)

    out["churn_risk"] = risk.fillna(0).astype(int)

    return out
