from math import log2

def get_recall_at_100(results_ids, pos_pas_ids):
    # get matching doc ids in results
    top_100_pos_pas = [r for r in results_ids[:100] if r in pos_pas_ids]

    # calculate recall
    recall_at_100_score = len(top_100_pos_pas) / len(pos_pas_ids)

    return recall_at_100_score

# def get_nDCG_at_10(results, pos_pas, neg_pas):
#     # get doc ids of positive and negative passages
#     pos_pas_ids = [p['docid'] for p in pos_pas]
#     neg_pas_ids = [p['docid'] for p in neg_pas]

#     # get matching doc ids in results
#     top_10_pos_pas = [r for r in results if r in pos_pas_ids]
#     top_10_neg_pas = [r for r in results if r in neg_pas_ids]

#     # check for zero division
#     if len(top_10_pos_pas) + len(top_10_neg_pas) == 0:
#         ndcg_at_10_score = 0
#         return ndcg_at_10_score

#     # calculate normalized discounted cumulative gain at 10
#     dcg_at_10_score = sum([1 / math.log(i + 2, 2) for i in range(len(top_10_pos_pas))])
#     idcg_at_10_score = sum([1 / math.log(i + 2, 2) for i in range(len(top_10_pos_pas) + len(top_10_neg_pas))])
#     ndcg_at_10_score = dcg_at_10_score / idcg_at_10_score

#     return ndcg_at_10_score


def get_nDCG_at_10(results_ids, pos_pas_ids, k=10):
    # Extract relevance scores for each document in the top k results
    relevance_scores = [1 if result in pos_pas_ids else 0 for result in results_ids[:k]]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = sum((2 ** rel - 1) / (log2(i + 2)) for i, rel in enumerate(relevance_scores))

    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    ideal_relevance_scores = [1] * min(k, len(pos_pas_ids))
    idcg = sum((2 ** rel - 1) / (log2(i + 2)) for i, rel in enumerate(ideal_relevance_scores))

    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg