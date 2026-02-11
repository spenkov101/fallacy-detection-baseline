from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
import pandas as pd

def _infer_key(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Pick the first key that exists in the example."""
    for k in candidates:
        if k in example:
            return k
    return None


def infer_schema(
    examples: List[Dict[str, Any]],
    text_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Infer likely text/label keys from the first non-empty example.
    Override by passing text_key/label_key explicitly.
    """
    if text_key and label_key:
        return text_key, label_key

    # Find first dict-like example
    ex0 = None
    for ex in examples:
        if isinstance(ex, dict) and ex:
            ex0 = ex
            break
    if ex0 is None:
        raise ValueError("No usable examples to infer schema from.")

    text_candidates = ["text", "sentence", "claim", "content", "premise"]
    label_candidates = ["label", "fallacy", "class", "gold", "y"]

    inferred_text = text_key or _infer_key(ex0, text_candidates)
    inferred_label = label_key or _infer_key(ex0, label_candidates)

    if not inferred_text:
        raise ValueError(f"Could not infer text key. Tried: {text_candidates}")
    if not inferred_label:
        raise ValueError(f"Could not infer label key. Tried: {label_candidates}")

    return inferred_text, inferred_label


def label_counts(
    examples: List[Dict[str, Any]],
    label_key: Optional[str] = None,
) -> Counter:
    """Count examples per label."""
    _, lk = infer_schema(examples, label_key=label_key)
    return Counter(ex.get(lk) for ex in examples)


def length_stats_by_label(
    examples: List[Dict[str, Any]],
    text_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> Dict[Any, Dict[str, float]]:
    """
    Compute simple token-length stats per label: count, mean, min, max.
    Tokenization = whitespace split (baseline-friendly).
    """
    tk, lk = infer_schema(examples, text_key=text_key, label_key=label_key)

    lengths = defaultdict(list)
    for ex in examples:
        text = str(ex.get(tk, "")).strip()
        label = ex.get(lk)
        if text == "":
            continue
        lengths[label].append(len(text.split()))

    out: Dict[Any, Dict[str, float]] = {}
    for label, vals in lengths.items():
        out[label] = {
            "count": float(len(vals)),
            "mean": sum(vals) / len(vals) if vals else 0.0,
            "min": float(min(vals)) if vals else 0.0,
            "max": float(max(vals)) if vals else 0.0,
        }
    return out


def dataset_sanity_report(
    examples: List[Dict[str, Any]],
    text_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick sanity report: missing/empty text, missing labels, inferred schema.
    """
    tk, lk = infer_schema(examples, text_key=text_key, label_key=label_key)

    missing_text = 0
    empty_text = 0
    missing_label = 0

    for ex in examples:
        if tk not in ex:
            missing_text += 1
        else:
            if str(ex.get(tk, "")).strip() == "":
                empty_text += 1
        if lk not in ex:
            missing_label += 1

    return {
        "num_examples": len(examples),
        "text_key": tk,
        "label_key": lk,
        "missing_text_key": missing_text,
        "empty_text": empty_text,
        "missing_label_key": missing_label,
        "label_counts": dict(Counter(ex.get(lk) for ex in examples)),
    }
    

PRONOUNS = {"i", "me", "my", "mine", "we", "our", "us"}


def stylistic_cue_profile(
    examples: List[Dict[str, Any]],
    text_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> Dict[Any, Dict[str, float]]:
    """
    Compute simple stylistic cue frequencies per label:
    - question marks
    - exclamation marks
    - ALL CAPS tokens
    - first-person pronouns

    Returns average count per example for each cue.
    """
    tk, lk = infer_schema(examples, text_key=text_key, label_key=label_key)

    stats = defaultdict(lambda: {"count": 0, "q_marks": 0, "ex_marks": 0, "all_caps": 0, "pronouns": 0})

    for ex in examples:
        text = str(ex.get(tk, ""))
        label = ex.get(lk)
        if not text:
            continue

        stats[label]["count"] += 1
        stats[label]["q_marks"] += text.count("?")
        stats[label]["ex_marks"] += text.count("!")

        tokens = re.findall(r"\b\w+\b", text)
        stats[label]["all_caps"] += sum(1 for t in tokens if t.isupper() and len(t) > 1)
        stats[label]["pronouns"] += sum(1 for t in tokens if t.lower() in PRONOUNS)

    # convert to averages per example
    out = {}
    for label, d in stats.items():
        c = d["count"] if d["count"] else 1
        out[label] = {
            "avg_question_marks": d["q_marks"] / c,
            "avg_exclamation_marks": d["ex_marks"] / c,
            "avg_all_caps_tokens": d["all_caps"] / c,
            "avg_first_person_pronouns": d["pronouns"] / c,
        }

    return out
    
ef run_basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience wrapper to run core analysis directly on a DataFrame.
    Assumes columns: 'text' and 'label'.
    """
    examples = df.to_dict(orient="records")

    return {
        "sanity": dataset_sanity_report(examples),
        "label_counts": dict(label_counts(examples)),
        "length_stats": length_stats_by_label(examples),
    }
