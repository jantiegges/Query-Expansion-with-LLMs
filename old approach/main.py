import psycopg2
from datasets import load_dataset
import psycopg2.extras
from tqdm import tqdm
from nltk.tokenize import word_tokenize

from utils.metrics import get_recall_at_100, get_nDCG_at_10

def execute_bm25_query(cur, query, with_scores=False):
    query = f"{query}:::limit=100"
    if with_scores:
        cur.execute("""
            SELECT *
            FROM corpusfr
            WHERE corpusfr @@@ %s;
            """, (query,))
    else:
        cur.execute("""
            SELECT *, paradedb.rank_bm25(ctid)
            FROM corpusfr
            WHERE corpusfr @@@ %s
            ORDER BY rank_bm25 DESC;
            """, (query,))

    return cur.fetchall()


def execute_similarity_query(cur, query_emb):
    # convert query_emb (list of floats) to be sql compatible
    emb_string = "[" + ",".join(map(str, query_emb)) + "]"

    cur.execute("""
        SELECT *
        FROM corpusfr
        ORDER BY corpusfr.emb <-> %s
        LIMIT 100;
        """, (emb_string,))
    return cur.fetchall()


def execute_hybrid_query(cur, query, query_emb, weights):
    emb_string = "[" + ",".join(map(str, query_emb)) + "]"

    cur.execute("""
        SELECT *,
        paradedb.weighted_mean(
            paradedb.minmax_bm25(ctid, 'idx_corpusfr', %s),
            1 - paradedb.minmax_norm(
                emb <-> %s,
                MIN(emb <-> %s) OVER (),
                MAX(emb <-> %s) OVER ()
            ),
            %s
        ) as score_hybrid
    FROM corpusfr
    ORDER BY score_hybrid DESC
    LIMIT 100;
    """, (query, emb_string, emb_string, emb_string, weights))

    return cur.fetchall()


def process_query_with_nltk(query):
    tokens = word_tokenize(query)
    tokens = [token.lower() for token in tokens if len(token.encode('utf-8')) <= 255]

    return tokens

if __name__ == "__main__":

    # the dev set has also been used in the original paper for evaluation
    dataset = load_dataset("Cohere/miracl-fr-queries-22-12", split="dev")

    # METHODOLOGY: Preprocess Query -> Apply Retrieval Method -> Apply Re-Ranking Method -> Select Results

    connection_parameters = {
        'host': "localhost",
        'port': 5432,
        'dbname': "comp550-final",
        'user': "postgres",
        'password': "password"
    }

    with psycopg2.connect(**connection_parameters) as conn:
        with conn.cursor() as cur:

            bm25_recall = []
            bm25_nDCG = []
            sim_recall = []
            sim_nDCG = []
            hybrid_recall = []
            hybrid_nDCG = []

            for data in tqdm(dataset, total=343):
                query_id = data['query_id']
                query = data['query']
                positive_passages = data['positive_passages']
                negative_passages = data['negative_passages']
                pos_pas_ids = [p['docid'] for p in positive_passages]

                query = process_query_with_nltk(query)

                # get query results
                bm25_results = execute_bm25_query(cur, query)
                bm25_results_ids = [r[0] for r in bm25_results]

                sim_results = execute_similarity_query(cur, data['emb'])
                sim_results_id = [r[0] for r in sim_results]

                hybrid_results = execute_hybrid_query(cur, query, data['emb'], [0.5, 0.5])
                hybrid_results_id = [r[0] for r in hybrid_results]

                # get recall@100 and nDCG@10 scores
                bm25_recall.append(get_recall_at_100(bm25_results_ids, pos_pas_ids))
                bm25_nDCG.append(get_nDCG_at_10(bm25_results_ids, pos_pas_ids))

                sim_recall.append(get_recall_at_100(sim_results_id, positive_passages, negative_passages))
                sim_nDCG.append(get_nDCG_at_10(sim_results_id, positive_passages, negative_passages))

                hybrid_recall.append(get_recall_at_100(hybrid_results_id, positive_passages, negative_passages))
                hybrid_nDCG.append(get_nDCG_at_10(hybrid_results_id, positive_passages, negative_passages))

            print("BM25 Recall@100: ", sum(bm25_recall) / len(bm25_recall))
            print("BM25 nDCG@10: ", sum(bm25_nDCG) / len(bm25_nDCG))
            print("Similarity Recall@100: ", sum(sim_recall) / len(sim_recall))
            print("Similarity nDCG@10: ", sum(sim_nDCG) / len(sim_nDCG))
            print("Hybrid Recall@100: ", sum(hybrid_recall) / len(hybrid_recall))
            print("Hybrid nDCG@10: ", sum(hybrid_nDCG) / len(hybrid_nDCG))