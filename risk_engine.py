def calculate_risk(speaker_similarity, spoof_score):
    """
    VoxGuard combined speaker verification + anti-spoofing
    risk engine.
    """

    # ---------------------------------------
    # Speaker verification
    # ---------------------------------------

    if speaker_similarity >= 0.50:
        speaker_status = "VERIFIED"
    elif speaker_similarity >= 0.30:
        speaker_status = "SUSPICIOUS"
    else:
        speaker_status = "MISMATCH"

    # ---------------------------------------
    # Spoof detection
    # ---------------------------------------

    if spoof_score >= 0.50:
        spoof_label = "spoof"
    else:
        spoof_label = "bonafide"

    # ---------------------------------------
    # Risk calculation
    # ---------------------------------------

    if speaker_status == "VERIFIED" and spoof_score >= 0.50:
        # Same speaker but AI/replay/spoof detected
        # This is the most dangerous scenario.
        final_risk_score = 90

    elif speaker_status == "MISMATCH" and spoof_score >= 0.50:
        # Different speaker + spoof
        final_risk_score = 95

    elif speaker_status == "SUSPICIOUS" and spoof_score >= 0.50:
        # Uncertain speaker + spoof
        final_risk_score = 85

    elif speaker_status == "MISMATCH":
        # Different genuine speaker
        final_risk_score = 80

    elif speaker_status == "SUSPICIOUS":
        # Uncertain speaker, apparently genuine audio
        final_risk_score = 50

    else:
        # Verified speaker + apparently genuine audio
        final_risk_score = round(spoof_score * 30)

    # ---------------------------------------
    # Risk level
    # ---------------------------------------

    if final_risk_score >= 70:
        risk_level = "HIGH"
    elif final_risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "speaker_similarity": round(speaker_similarity, 4),
        "speaker_status": speaker_status,
        "spoof_score": round(spoof_score, 4),
        "spoof_label": spoof_label,
        "final_risk_score": final_risk_score,
        "risk_level": risk_level,
    }


# ---------------------------------------
# Test
# ---------------------------------------

if __name__ == "__main__":

    result = calculate_risk(
        speaker_similarity=0.72,
        spoof_score=0.85
    )

    print(result)