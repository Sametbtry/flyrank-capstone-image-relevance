SIM_THRESHOLD = 0.33
CONF_THRESHOLD = 0.70


def check_mismatch_guard(
    image_category: str,
    expected_category: str | None,
    similarity: float,
    confidence: float,
) -> tuple[bool, str | None]:
    """Three-check mismatch guard. Returns (passed, rejection_reason)."""
    if expected_category is None or image_category != expected_category:
        return (
            False,
            f"Subject mismatch: expected {expected_category}, detected {image_category}",
        )

    if similarity < SIM_THRESHOLD:
        return (
            False,
            "Similarity score below threshold: image content is not relevant enough",
        )

    if confidence < CONF_THRESHOLD:
        return (
            False,
            "Vision model confidence too low to accept this image safely",
        )

    return True, None
