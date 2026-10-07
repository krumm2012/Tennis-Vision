"""Versioned amateur practice scoring; projection estimates and coach reviews stay separate.

This module and practice_policy.json are vendored identically in Tennis-Vision.
No cross-repository import or implicit historical score/scale conversion is used.
"""
import hashlib
import json
import math
from pathlib import Path

POLICY_PATH = Path(__file__).with_name("practice_policy.json")
POLICY = json.loads(POLICY_PATH.read_text())
POLICY_VERSION = POLICY["version"]
POLICY_SHA256 = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
DIMENSIONS = tuple(d["id"] for d in POLICY["dimensions"])


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def grade(score):
    score = number(score)
    if score is None or not 0 <= score <= 100:
        return None
    return next(code for lower, code, _ in POLICY["grade_bands"] if score >= lower)


def score_review(review):
    """All five explicit ratings and confirmation are required for a total."""
    if not isinstance(review, dict):
        raise ValueError("Practice review must be an object")
    ratings = review.get("ratings")
    if ratings is None:
        ratings = {}
    if not isinstance(ratings, dict):
        raise ValueError("Ratings must be an object")
    if set(ratings) - set(DIMENSIONS):
        raise ValueError("Unknown practice dimension")
    values = {}
    for key in DIMENSIONS:
        value = ratings.get(key)
        if value is not None and (type(value) is not int or not 1 <= value <= 5):
            raise ValueError("Ratings must be integers 1–5 or null")
        values[key] = value
    used = sum(v is not None for v in values.values())
    confirmed = review.get("confirmed") is True
    complete = used == len(DIMENSIONS)
    score = round(sum(values.values()) / (5 * len(DIMENSIONS)) * 100, 1) if complete and confirmed else None
    return {
        "policy_version": POLICY_VERSION, "policy_sha256": POLICY_SHA256,
        "method": "coach_manual", "scope": POLICY["manual_scope"], "scale": [0, 100],
        "status": "coach_confirmed" if score is not None else "draft",
        "score": score, "grade": grade(score), "coverage": used / len(DIMENSIONS),
        "confidence": None, "ratings": values, "validation_status": POLICY["status"],
    }


def score_event(event):
    if event.get("practice_review"):
        reviewed = score_review(event["practice_review"])
        if reviewed["status"] == "coach_confirmed":
            # Retain automatic evidence alongside the separate manual series.
            reviewed["calibration"] = score_event({k: v for k, v in event.items() if k != "practice_review"})["calibration"]
            return reviewed
    # Import only in the analyzer adapter; Viewer uses score_review without this dependency.
    from swing_coach_calibration import calibrate_coaching_event
    calibration = calibrate_coaching_event(event)
    visible = number(calibration.get("visible_technique_score"))
    score = round(visible * 100, 1) if visible is not None else None
    return {
        "policy_version": POLICY_VERSION, "policy_sha256": POLICY_SHA256,
        "method": "automatic_2d_projection", "scope": POLICY["automatic_scope"], "scale": [0, 100],
        "status": "provisional" if score is not None else calibration["status"],
        "score": score, "grade": grade(score),
        "coverage": len(calibration["metrics_used"]) / 3,
        "confidence": calibration["confidence"], "calibration": calibration,
        "validation_status": POLICY["status"],
    }


def attach_score(event):
    result = score_event(event)
    event["practice_score"] = result
    event["coach_calibration"] = result["calibration"]
    event["swing_score"] = result["score"]
    event["swing_grade"] = result["grade"]
    bio = event.get("biomechanics") or {}
    event["biomechanics"] = bio
    bio.update(practice_score=result, swing_score=result["score"], swing_grade=result["grade"])
    ext = bio.get("extended_biomechanics") or {}
    existing_sub = ext.get("swing_quality_score", {}).get("sub_scores") or (bio.get("metrics", {}).get("swing_quality_score", {}) or {}).get("sub_scores") or {}
    score = {"overall_score": result["score"], "grade": result["grade"], "sub_scores": existing_sub,
             "policy_version": POLICY_VERSION, "scope": result["scope"]}
    bio["extended_biomechanics"] = ext
    event["extended_biomechanics"] = ext
    ext["swing_quality_score"] = score
    if "metrics" in bio:
        bio["metrics"]["swing_quality_score"] = {"value": result["score"], "grade": result["grade"],
             "confidence": result["confidence"], "coach_eligible": False, "sub_scores": existing_sub,
             "exclusion_reason": "score_is_not_an_independent_measurement"}
    return result
