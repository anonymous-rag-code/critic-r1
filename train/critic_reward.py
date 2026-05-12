
import os
import re
import time
import random
import ast
import math
from collections import Counter
from typing import Any, Dict, Tuple, List




def resolve_training_steps(**kwargs):
    """
    Resolve reward-stage switches from kwargs or environment variables.

    STEP1=True enables conservative judgment alignment rewards.
    STEP2=True enables diagnostic quality alignment rewards.
    """

    def parse_bool(x, default):
        if x is None:
            return default
        if isinstance(x, bool):
            return x
        s = str(x).lower().strip()
        return s in ("1", "true", "yes", "y")

    step1 = parse_bool(kwargs.get("step1", os.environ.get("STEP1")), True)
    step2 = parse_bool(kwargs.get("step2", os.environ.get("STEP2")), False)
    return step1, step2


# Conservative judgment alignment reward matrix.
# Rows are reference verdicts and columns are predicted verdicts.
# The matrix penalizes over-aggressive incorrect judgments more strongly
# than cautious uncertainty.
VERDICT_REWARD_MATRIX = {
    "CORRECT": {
        "CORRECT": 0.7,
        "INCORRECT": -1,
        "UNSURE": -0.1,
    },
    "INCORRECT": {
        "CORRECT": -0.3,
        "INCORRECT": 0.5,
        "UNSURE": -0.1,
    },
    "UNSURE": {
        "CORRECT": 0.1,
        "INCORRECT": -0.2,
        "UNSURE": 0.0,
    },
}

_TAGS = ["verdict", "location", "reason", "fix"]


# -----------------------------
# Basic extraction
# -----------------------------
def extract_tag_content(text: str, tag: str) -> str:
    if text is None:
        return ""
    m = re.search(
        fr"<{tag}>\s*(.*?)\s*</{tag}>",
        str(text),
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def extract_verdict(text: str) -> str:
    v = extract_tag_content(text, "verdict").strip().upper()
    if v in ("CORRECT", "INCORRECT", "UNSURE"):
        return v
    return ""


def extract_location(text: str) -> str:
    return extract_tag_content(text, "location")


def extract_reason(text: str) -> str:
    return extract_tag_content(text, "reason")


def extract_fix(text: str) -> str:
    return extract_tag_content(text, "fix")




def split_phrases(text: str) -> List[str]:
    s = "" if text is None else str(text).strip().lower()
    if not s:
        return []
    parts = [x.strip() for x in s.split(",") if x.strip()]
    seen = set()
    out = []
    for p in parts:
        p2 = re.sub(r"\s+", " ", p)
        if p2 not in seen:
            seen.add(p2)
            out.append(p2)
    return out


def normalize_text_for_match(text: str) -> str:
    s = "" if text is None else str(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Lexical similarity helpers
# -----------------------------
def token_f1_score(pred_text: str, gold_text: str) -> float:
    pred = normalize_text_for_match(pred_text)
    gold = normalize_text_for_match(gold_text)

    if not pred or not gold:
        return 0.0

    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / max(len(pred_tokens), 1)
    recall = num_same / max(len(gold_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def compute_keywords_coverage_score(
    pred_text: str,
    standard_keywords: str,
) -> Tuple[float, List[str]]:
    pred_norm = normalize_text_for_match(pred_text)
    gold_keywords = split_phrases(standard_keywords)

    if not pred_norm or not gold_keywords:
        return 0.0, []

    matched = []
    seen = set()

    for kw in gold_keywords:
        kw_norm = normalize_text_for_match(kw)
        if kw_norm and kw_norm in pred_norm and kw_norm not in seen:
            seen.add(kw_norm)
            matched.append(kw)

    coverage = len(matched) / max(len(gold_keywords), 1)
    coverage = max(0.0, min(1.0, coverage))
    return coverage, matched


def normalized_exponential_reward(
    score: float,
    max_reward: float,
    beta: float = 4.0,
) -> float:
    """
    score in [0,1]
    return in [0,max_reward]

    R = max_reward * (exp(beta * score) - 1) / (exp(beta) - 1)
    """
    s = max(0.0, min(1.0, float(score)))
    denom = math.exp(beta) - 1.0
    if denom <= 1e-12:
        return max_reward * s
    return max_reward * (math.exp(beta * s) - 1.0) / denom


# -----------------------------
# Read supervision fields
# -----------------------------
def _collect_candidate_containers(kwargs: Dict[str, Any]):
    candidates = []

    if "extra_info" in kwargs:
        candidates.append(kwargs.get("extra_info"))
    if "sample" in kwargs:
        candidates.append(kwargs.get("sample"))
    if "data" in kwargs:
        candidates.append(kwargs.get("data"))
    if "item" in kwargs:
        candidates.append(kwargs.get("item"))
    if "batch" in kwargs:
        candidates.append(kwargs.get("batch"))

    return candidates


def get_standard_critique(**kwargs) -> str:
    for c in _collect_candidate_containers(kwargs):
        if not isinstance(c, dict):
            continue

        if "standard_critique" in c and c.get("standard_critique"):
            return str(c.get("standard_critique"))

        ei = c.get("extra_info")
        if isinstance(ei, dict) and ei.get("standard_critique"):
            return str(ei.get("standard_critique"))

    return ""


def get_standard_verdict(**kwargs) -> str:
    sc = get_standard_critique(**kwargs)
    return extract_verdict(sc)


def get_standard_location(**kwargs) -> str:
    sc = get_standard_critique(**kwargs)
    return extract_location(sc)


def get_standard_reason(**kwargs) -> str:
    for c in _collect_candidate_containers(kwargs):
        if not isinstance(c, dict):
            continue

        if "standard_reason" in c and c.get("standard_reason"):
            return str(c.get("standard_reason"))

        ei = c.get("extra_info")
        if isinstance(ei, dict) and ei.get("standard_reason"):
            return str(ei.get("standard_reason"))

    sc = get_standard_critique(**kwargs)
    return extract_reason(sc)


def get_standard_fix(**kwargs) -> str:
    for c in _collect_candidate_containers(kwargs):
        if not isinstance(c, dict):
            continue

        if "standard_fix" in c and c.get("standard_fix"):
            return str(c.get("standard_fix"))

        ei = c.get("extra_info")
        if isinstance(ei, dict) and ei.get("standard_fix"):
            return str(ei.get("standard_fix"))

    sc = get_standard_critique(**kwargs)
    return extract_fix(sc)


def get_standard_keywords(**kwargs) -> str:
    for c in _collect_candidate_containers(kwargs):
        if not isinstance(c, dict):
            continue

        if "keywords" in c and c.get("keywords"):
            return str(c.get("keywords"))

        ei = c.get("extra_info")
        if isinstance(ei, dict) and ei.get("keywords"):
            return str(ei.get("keywords"))

    return ""



# -----------------------------
# Critic format checking
# -----------------------------
def _count_tag(text: str, tag: str) -> Tuple[int, int]:
    open_c = len(re.findall(fr"<{tag}>", text, flags=re.IGNORECASE))
    close_c = len(re.findall(fr"</{tag}>", text, flags=re.IGNORECASE))
    return open_c, close_c


def is_valid_critic_format(text: str) -> Tuple[bool, str]:
    """
    Validate critic output format:
      - Must start with <verdict> (ignoring leading whitespace/newlines)
      - Must contain exactly one pair of each tag: verdict/location/reason/fix
      - Must not have extra non-whitespace content outside the four-tag blob
    """
    if text is None:
        return False, "Empty output"

    t = str(text)

    if not t.lstrip().lower().startswith("<verdict>"):
        return False, "Does not start with <verdict>"

    for tag in _TAGS:
        oc, cc = _count_tag(t, tag)
        if oc != 1 or cc != 1:
            return False, f"Tag count mismatch for {tag}: open={oc}, close={cc}"

    pattern = (
        r"^\s*"
        r"<verdict>.*?</verdict>\s*"
        r"<location>.*?</location>\s*"
        r"<reason>.*?</reason>\s*"
        r"<fix>.*?</fix>\s*"
        r"$"
    )
    if not re.match(pattern, t, flags=re.DOTALL | re.IGNORECASE):
        return False, "Extra content outside tags or wrong tag order"

    return True, "OK"


# -----------------------------
# Verdict reward
# -----------------------------
def compute_verdict_reward(pred_verdict: str, standard_verdict: str) -> float:
    if not pred_verdict or not standard_verdict:
        return -0.5
    return VERDICT_REWARD_MATRIX.get(standard_verdict, {}).get(pred_verdict, -0.5)


# -----------------------------
# Location parsing + reward
# -----------------------------
def parse_location(loc: str) -> Dict[str, Any]:
    s = "" if loc is None else str(loc).strip().lower()
    s = re.sub(r"\s+", "", s)

    if not s:
        return {"kind": "", "index": None, "raw": s}

    if s == "none":
        return {"kind": "none", "index": None, "raw": s}

    if s == "answer":
        return {"kind": "answer", "index": None, "raw": s}

    if s.startswith("think"):
        m = re.search(r"step(\d+|k)", s)
        idx = m.group(1) if m else None
        return {"kind": "think", "index": idx, "raw": s}

    if s.startswith("search"):
        m = re.search(r"step(\d+|k)", s)
        idx = m.group(1) if m else None
        return {"kind": "search", "index": idx, "raw": s}

    if "information" in s or re.search(r"doc(\d+|k)", s):
        m = re.search(r"doc(\d+|k)", s)
        idx = m.group(1) if m else None
        return {"kind": "information", "index": idx, "raw": s}

    return {"kind": "other", "index": None, "raw": s}


def compute_location_reward(
    pred_location: str,
    standard_location: str,
    location_type_score: float = 0.35,
    location_index_score: float = 0.20,
) -> float:
    pred = parse_location(pred_location)
    gold = parse_location(standard_location)

    if not pred["kind"] or not gold["kind"]:
        return 0.0

    if pred["kind"] != gold["kind"]:
        return 0.0

    reward = location_type_score

    if gold["kind"] in ("none", "answer"):
        return reward + location_index_score

    if pred["index"] is not None and gold["index"] is not None:
        if pred["index"] == gold["index"]:
            reward += location_index_score

    return reward


# -----------------------------
# Reason reward: F1 similarity
# -----------------------------
def compute_reason_f1_reward(
    pred_reason: str,
    standard_reason: str,
    max_total_reward: float = 0.5,
    exp_beta: float = 4.0,
) -> Tuple[float, float]:
    f1 = token_f1_score(pred_reason, standard_reason)
    reward = normalized_exponential_reward(
        score=f1,
        max_reward=max_total_reward,
        beta=exp_beta,
    )
    return reward, f1


# -----------------------------
# Fix reward: F1 + keywords
# -----------------------------
def compute_fix_hybrid_reward(
    pred_fix: str,
    standard_fix: str,
    standard_keywords: str,
    max_total_reward: float = 0.45,
    f1_weight: float = 0.7,
    keyword_weight: float = 0.3,
    exp_beta: float = 4.0,
) -> Tuple[float, Dict[str, Any]]:
    fix_f1 = token_f1_score(pred_fix, standard_fix)

    keyword_score, matched_keywords = compute_keywords_coverage_score(
        pred_text=pred_fix,
        standard_keywords=standard_keywords,
    )

    mix_score = f1_weight * fix_f1 + keyword_weight * keyword_score
    mix_score = max(0.0, min(1.0, mix_score))

    reward = normalized_exponential_reward(
        score=mix_score,
        max_reward=max_total_reward,
        beta=exp_beta,
    )

    info = {
        "fix_f1": fix_f1,
        "keyword_score": keyword_score,
        "mixed_score": mix_score,
        "matched_keywords": matched_keywords,
    }
    return reward, info


def compute_fix_generic_penalty(
    pred_fix: str,
    penalty: float = -0.02,
) -> float:
    s = normalize_text_for_match(pred_fix)
    generic = {
        "search more",
        "search again",
        "find evidence",
        "search better",
        "look up again",
        "search for more information",
    }
    return penalty if s in generic else 0.0


# -----------------------------
# Trivial / lazy output penalty
# -----------------------------
def word_count(text: str) -> int:
    s = "" if text is None else str(text).strip()
    if not s:
        return 0
    return len(re.findall(r"\S+", s))


def compute_trivial_penalty(
    pred_location: str,
    standard_location: str,
    reason: str,
    fix: str,
    none_penalty: float = -0.05,
    short_reason_penalty: float = -0.05,
    short_fix_penalty: float = -0.05,
    min_reason_words: int = 2,
    min_fix_words: int = 2,
    enable_location_penalty: bool = True,
    enable_short_reason_penalty: bool = True,
    enable_short_fix_penalty: bool = True,
) -> float:
    penalty = 0.0

    pred_loc = parse_location(pred_location)
    gold_loc = parse_location(standard_location)

    if enable_location_penalty:
        if gold_loc["kind"] and gold_loc["kind"] != "none" and pred_loc["kind"] == "none":
            penalty += none_penalty

    if enable_short_reason_penalty:
        if word_count(reason) < min_reason_words:
            penalty += short_reason_penalty

    if enable_short_fix_penalty:
        if word_count(fix) < min_fix_words:
            penalty += short_fix_penalty

    return penalty


# -----------------------------
# Scoring function
# -----------------------------
def compute_score_critic(
    solution_str: str,
    ground_truth: Dict[str, Any], # kept for VERL reward interface compatibility
    format_score: float = 0.2,
    location_type_score: float = 0.35,
    location_index_score: float = 0.20,
    reason_max_reward: float = 0.5,
    fix_max_reward: float = 0.45,
    generic_fix_penalty_value: float = -0.02,
    none_penalty: float = -0.05,
    short_reason_penalty: float = -0.05,
    short_fix_penalty: float = -0.05,
    min_reason_words: int = 2,
    min_fix_words: int = 2,
    debug_prob: int = 0,
    structure_format_score: float = None,
    **kwargs,
) -> float:
    """
    Compute the structured critic reward.

    The reward consists of:
    - format reward
    - conservative verdict reward
    - gated diagnostic rewards for location, reason, and fix
    - penalties for trivial or generic outputs
    """
    if structure_format_score is not None:
        format_score = structure_format_score

    step1, step2 = resolve_training_steps(**kwargs)

    reason_exp_beta = float(kwargs.get("reason_exp_beta", 4.0))
    fix_exp_beta = float(kwargs.get("fix_exp_beta", 4.0))
    fix_f1_weight = float(kwargs.get("fix_f1_weight", 0.7))
    fix_keyword_weight = float(kwargs.get("fix_keyword_weight", 0.3))

    # 1) format
    valid_fmt, fmt_msg = is_valid_critic_format(solution_str)

    # 2) parse model prediction
    pred_verdict = extract_verdict(solution_str) if valid_fmt else ""
    pred_location = extract_location(solution_str) if valid_fmt else ""
    pred_reason = extract_reason(solution_str) if valid_fmt else ""
    pred_fix = extract_fix(solution_str) if valid_fmt else ""

    # 3) parse supervision
    standard_verdict = get_standard_verdict(**kwargs)
    standard_location = get_standard_location(**kwargs)
    standard_reason = get_standard_reason(**kwargs)
    standard_fix = get_standard_fix(**kwargs)
    standard_keywords = get_standard_keywords(**kwargs)


    # 4) base rewards
    fmt_reward = format_score if valid_fmt else -2.0

    correctness_reward = 0.0
    if valid_fmt and standard_verdict:
        correctness_reward = compute_verdict_reward(pred_verdict, standard_verdict)

    # Only when verdict is correct, compute downstream rewards/penalties
    verdict_is_correct = bool(valid_fmt and standard_verdict and pred_verdict == standard_verdict)

    location_reward = 0.0
    reason_reward = 0.0
    fix_reward = 0.0
    fix_penalty = 0.0
    trivial_penalty = 0.0

    reason_f1 = 0.0
    fix_reward_info: Dict[str, Any] = {}

    if verdict_is_correct:
        has_location_supervision = bool(standard_location and str(standard_location).strip())
        has_reason_supervision = bool(standard_reason and str(standard_reason).strip())
        has_fix_supervision = bool(
            (standard_fix and str(standard_fix).strip()) or
            (standard_keywords and str(standard_keywords).strip())
        )

        # Step1
        if step1 and standard_verdict in ("INCORRECT", "UNSURE"):
            trivial_penalty = compute_trivial_penalty(
                pred_location=pred_location,
                standard_location=standard_location,
                reason=pred_reason,
                fix=pred_fix,
                none_penalty=none_penalty,
                short_reason_penalty=short_reason_penalty,
                short_fix_penalty=short_fix_penalty,
                min_reason_words=min_reason_words,
                min_fix_words=min_fix_words,
                enable_location_penalty=False,
                enable_short_reason_penalty=has_reason_supervision,
                enable_short_fix_penalty=has_fix_supervision,
            )

        # Step2
        if step2 and standard_verdict in ("CORRECT", "INCORRECT", "UNSURE"):
            if has_location_supervision:
                location_reward = compute_location_reward(
                    pred_location=pred_location,
                    standard_location=standard_location,
                    location_type_score=location_type_score,
                    location_index_score=location_index_score,
                )

            if has_reason_supervision:
                reason_reward, reason_f1 = compute_reason_f1_reward(
                    pred_reason=pred_reason,
                    standard_reason=standard_reason,
                    max_total_reward=reason_max_reward,
                    exp_beta=reason_exp_beta,
                )

            if has_fix_supervision:
                fix_reward, fix_reward_info = compute_fix_hybrid_reward(
                    pred_fix=pred_fix,
                    standard_fix=standard_fix,
                    standard_keywords=standard_keywords,
                    max_total_reward=fix_max_reward,
                    f1_weight=fix_f1_weight,
                    keyword_weight=fix_keyword_weight,
                    exp_beta=fix_exp_beta,
                )

            trivial_penalty = compute_trivial_penalty(
                pred_location=pred_location,
                standard_location=standard_location,
                reason=pred_reason,
                fix=pred_fix,
                none_penalty=none_penalty,
                short_reason_penalty=short_reason_penalty,
                short_fix_penalty=short_fix_penalty,
                min_reason_words=min_reason_words,
                min_fix_words=min_fix_words,
                enable_location_penalty=has_location_supervision,
                enable_short_reason_penalty=has_reason_supervision,
                enable_short_fix_penalty=has_fix_supervision,
            )

            if has_fix_supervision:
                fix_penalty = compute_fix_generic_penalty(
                    pred_fix=pred_fix,
                    penalty=generic_fix_penalty_value,
                )

    total = (
        fmt_reward
        + correctness_reward
        + location_reward
        + reason_reward
        + fix_reward
        + trivial_penalty
        + fix_penalty
    )

    if debug_prob is not None and debug_prob > 0 and random.randint(1, debug_prob) == 1:
        print("----- critic reward debug -----")
        print("step1:", step1, "step2:", step2)
        print("standard verdict:", standard_verdict)
        print("pred verdict:", pred_verdict)
        print("verdict_is_correct:", verdict_is_correct)
        print("reason_f1:", reason_f1)
        print("fix reward info:", fix_reward_info)
        print("format ok:", valid_fmt, "msg:", fmt_msg)
        print("format reward:", fmt_reward)
        print("correctness reward:", correctness_reward)
        print("location reward:", location_reward)
        print("reason reward:", reason_reward)
        print("fix_reward:", fix_reward)
        print("trivial penalty:", trivial_penalty)
        print("fix_penalty:", fix_penalty)
        print("score:", total)


    return float(total)


# VERL reward entry point.
def compute_score(solution_str, ground_truth, **kwargs):
    return compute_score_critic(solution_str, ground_truth, **kwargs)


# Backward-compatible alias.
def compute_score_em(solution_str, ground_truth, **kwargs):
    return compute_score(solution_str, ground_truth, **kwargs)
    