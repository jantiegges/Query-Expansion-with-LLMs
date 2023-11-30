from math import log2

def get_recall_at_100(results_ids, pos_pas_ids):
    # get matching doc ids in results
    top_100_pos_pas = [r for r in results_ids[:100] if r in pos_pas_ids]

    recall_at_100_score = len(top_100_pos_pas) / len(pos_pas_ids) if len(pos_pas_ids) > 0 else 0

    return recall_at_100_score


def get_nDCG_at_10(results_ids, pos_pas_ids):
    # extract relevance scores
    relevance_scores = [1 if result in pos_pas_ids else 0 for result in results_ids[:10]]

    # calc Discounted Cumulative Gain (DCG)
    dcg = sum((2 ** rel - 1) / (log2(i + 2)) for i, rel in enumerate(relevance_scores))

    # calc Ideal Discounted Cumulative Gain (IDCG)
    ideal_relevance_scores = [1] * min(10, len(pos_pas_ids))
    idcg = sum((2 ** rel - 1) / (log2(i + 2)) for i, rel in enumerate(ideal_relevance_scores))

    # calc nDCG
    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg