def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Top-k recommendations
    top_k = recommended[:k]

    # Convert relevant items to set for fast lookup
    relevant_set = set(relevant)

    # Count hits
    hits = sum(item in relevant_set for item in top_k)

    # Compute metrics
    precision = hits / k
    recall = hits / len(relevant_set)

    return [float(precision), float(recall)]