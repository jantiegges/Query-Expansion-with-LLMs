from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from pyserini.search.hybrid import HybridSearcher
from datasets import load_dataset
from tqdm import tqdm
import time

from utils.metrics import get_recall_at_100, get_nDCG_at_10

def run_search(searcher, dataset):
    recall, ndcg = [], []
    for data in tqdm(dataset, total=len(dataset)):
        query = data['query']
        positive_passages = data['positive_passages']
        pos_pas_ids = [p['docid'] for p in positive_passages]

        # get query results
        results = searcher.search(query, k=100)
        results_ids = [r.docid for r in results]

        # get recall@100 and nDCG@10 scores
        recall.append(get_recall_at_100(results_ids, pos_pas_ids))
        ndcg.append(get_nDCG_at_10(results_ids, pos_pas_ids))

    recall = sum(recall) / len(recall)
    ndcg = sum(ndcg) / len(ndcg)

    return recall, ndcg


if __name__ == "__main__":
    ### EXPERIMENT 1: BASELINE ###
    print('EXPERIMENT 1: BASELINE')
    print("\n")
    dataset = load_dataset("Cohere/miracl-en-queries-22-12", split="dev")

    # BM25
    searcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    recall, ndcg = run_search(searcher, dataset)
    print(f'BM25 Recall@100: {recall:.4f}')
    print(f'BM25 nDCG@10: {ndcg:.4f}')
    print("\n")

    # mDPR
    encoder = TctColBertQueryEncoder('castorini/mdpr-tied-pft-msmarco')
    searcher = FaissSearcher.from_prebuilt_index(
        'miracl-v1.0-en-mdpr-tied-pft-msmarco',
        encoder
    )
    recall, ndcg = run_search(searcher, dataset)
    print(f'mDPR Recall@100: {recall:.4f}')
    print(f'mDPR nDCG@10: {ndcg:.4f}')
    print("\n")

    # Hybrid
    ssearcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    encoder = TctColBertQueryEncoder('castorini/mdpr-tied-pft-msmarco')
    dsearcher = FaissSearcher.from_prebuilt_index(
        'miracl-v1.0-en-mdpr-tied-pft-msmarco',
        encoder
    )
    hsearcher = HybridSearcher(dsearcher, ssearcher)
    recall, ndcg= run_search(hsearcher, dataset)
    print(f'Hybrid Recall@100: {recall:.4f}')
    print(f'Hybrid nDCG@10: {ndcg:.4f}')
    print("\n")


    ### EXPERIMENT 2: QUERY EXPANSION - ZERO SHOT ###

    ### EXPERIMENT 3: QUERY EXPANSION - ONE/MULTI SHOT ###

    ### EXPERIMENT 4: QUERY EXPANSION + RE-RANKING ###

    ### EXPERIMENT 5: QUERY EXPANSION - DIFFERENT LANGUAGE ###