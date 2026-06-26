def should_register_model(metrics: dict, params: dict) -> bool:
    """
    Decide whether the trained model should be registered.
    """

    metric_name = params["promotion"]["metric"]
    threshold = params["promotion"]["threshold"]

    score = metrics[metric_name]

    print(f"\nPromotion Metric : {metric_name}")
    print(f"Model Score      : {score:.4f}")
    print(f"Threshold        : {threshold:.4f}")

    return score >= threshold