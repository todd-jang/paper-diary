"""S_V: Vision anomaly agent.

- RT-DETR: 객체 위험도
- MAE siren: 사이렌 감지 (AST + template)
"""
from __future__ import annotations

from src.exp_a.vision_risk import anomaly_score as rtdetr_anomaly_score

try:
    from src.exp_a.mae_siren import mae_siren_score, siren_detected
    _HAS_MAE_SIREN = True
except Exception as e:
    print(f"[S_V] mae_siren import failed: {e}")
    _HAS_MAE_SIREN = False

    def mae_siren_score(state):
        return None

    def siren_detected(score, mode="balanced", level="clip"):
        return False

UI_FAST = True


def s_v_fn(state: dict) -> dict:
    # 1. vision anomaly (RT-DETR)
    try:
        rtdetr_score = rtdetr_anomaly_score(state)
    except Exception as e:
        print(f"[S_V] rtdetr failed: {e}")
        rtdetr_score = 0.0

    # 2. siren score (MAE + AST)
    siren_score = None
    siren_flag = False
    if _HAS_MAE_SIREN:
        try:
            siren_score = mae_siren_score(state)
            if siren_score is not None:
                siren_flag = siren_detected(siren_score, mode="balanced", level="clip")
        except Exception as e:
            print(f"[S_V] mae_siren failed: {e}")
            siren_score = None
            siren_flag = False

    # 3. metrics
    me = {
        "metrics": {
            "jitter_rms": 0,
            "steering_smoothness_dtheta_dt_rms": 0,
            "path_curvature_rms": 0,
            "event_alignment_ms": 0,
            "direction_match_rate": 0,
            "direction_mention_rate": 0,
        }
    }
    if UI_FAST:
        me["golden_fixtures"] = {
            "multi_frame_required": True,
            "sequences": ["seq_lead_lost_15f"],
            "ui_skip_check": True,
        }
    else:
        me["golden_fixtures"] = {
            "multi_frame_required": True,
            "sequences": ["seq_lead_lost_15f"],
        }

    # 4. event_trigger.details (s_v가 새로 만듦)
    event_trigger = state.get("event_trigger", {}) or {}
    # 기존 details 초기화 (이전 필드 제거)
    new_details = {
        "vision_anomaly_score": round(rtdetr_score, 4),
        "siren_score_mae": round(siren_score, 4) if siren_score is not None else None,
        "siren_detected_mae": siren_flag,
    }
    event_trigger["details"] = new_details

    return {
        "meta_exec": me,
        "event_trigger": event_trigger,
    }
