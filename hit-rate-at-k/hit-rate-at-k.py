def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    num_users = len(recommendations)

    # Return 0.0 if no users
    if num_users == 0:
        return 0.0

    hits = 0

    # Process each user
    for recs, truth in zip(recommendations, ground_truth):

        # Top-K recommendations
        top_k = set(recs[:k])

        # Relevant items
        relevant = set(truth)

        # Check if intersection is non-empty
        if top_k & relevant:
            hits += 1

    # Hit Rate
    hr = hits / num_users

    return float(hr)