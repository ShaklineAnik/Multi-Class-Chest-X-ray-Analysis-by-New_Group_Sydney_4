def confidence_level(confidence):
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "moderate"
    return "low"


def generate_research_explanation(prediction):
    """
    Lightweight LLM-style explanation for VS Code/local Flask deployment.
    This does not replace the Qwen LLM from Colab, but it uses the same structured inputs:
    predicted class, confidence, probability table, Grad-CAM result, and class labels.
    """
    pred_class = prediction["pred_class"]
    confidence = prediction["confidence"]
    level = confidence_level(confidence)
    probs = prediction["probabilities"]

    top_1 = probs[0]
    top_2 = probs[1]
    gap = top_1["probability"] - top_2["probability"]

    if gap < 0.10:
        uncertainty_note = (
            f"The probability gap between {top_1['class_name']} and {top_2['class_name']} is small, "
            "so the result should be interpreted cautiously."
        )
    else:
        uncertainty_note = (
            f"The model assigns a clearer probability margin over the second option, {top_2['class_name']}."
        )

    explanation = (
        f"The EfficientNetV2-S model predicts this chest X-ray as {pred_class} with {level} confidence "
        f"({confidence * 100:.2f}%). {uncertainty_note} The Grad-CAM visualization highlights the image regions "
        "that contributed most strongly to the CNN decision, helping users inspect whether the model focused on relevant chest/lung areas. "
        "This output is intended for research and decision-support demonstration only; it is not a standalone medical diagnosis and should be reviewed by a qualified clinician."
    )
    return explanation
