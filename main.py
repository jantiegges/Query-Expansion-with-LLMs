from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from pyserini.search.hybrid import HybridSearcher
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import pickle
import copy

from utils.metrics import get_recall_at_100, get_nDCG_at_10
from prompts import ZERO_SHOT_PROMPT, ONE_SHOT_PROMPT, MULTI_SHOT_PROMPT, ANSWER_PROMPT


# set up LLM
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

def get_query_expansion_dataset(dataset, option='zero-shot'):
    # try to load the expanded dataset
    try:
        with open(f'./data/miracl-en-queries-22-12-expanded-{option}.pkl', 'rb') as f:
            expanded_dataset = pickle.load(f)
        return expanded_dataset
    except:
        expanded_dataset = copy.deepcopy(dataset)
        for i in tqdm(range(len(dataset)), total=len(dataset), desc='Expanding queries'):
            query = dataset[i]['query']

            if option == 'zero-shot':
                messages = ZERO_SHOT_PROMPT.format_messages(query=query)
            elif option == 'one-shot':
                messages = ONE_SHOT_PROMPT.format_messages(query=query)
            elif option == 'multi-shot':
                messages = MULTI_SHOT_PROMPT.format_messages(query=query)
            elif option == 'answer':
                messages = ANSWER_PROMPT.format_messages(question=query)
            else:
                raise ValueError(f'Invalid option: {option}')

            expanded_query = chat(messages).content
            expanded_dataset[i]['query'] = expanded_query
            #print(expanded_dataset[i]['query'])
        
        with open(f'./data/miracl-en-queries-22-12-expanded-{option}.pkl', 'wb') as f:
            pickle.dump(expanded_dataset, f)

        return expanded_dataset

def run_search(searcher, dataset, k=100):
    recall, ndcg = [], []
    for data in tqdm(dataset, total=len(dataset), desc='Searching'):
        query = data['query']
        positive_passages = data['positive_passages']
        pos_pas_ids = [p['docid'] for p in positive_passages]

        # get query results
        results = searcher.search(query, k=k)
        results_ids = [r.docid for r in results]

        # get recall@100 and nDCG@10 scores
        recall.append(get_recall_at_100(results_ids, pos_pas_ids))
        ndcg.append(get_nDCG_at_10(results_ids, pos_pas_ids))

    recall = sum(recall) / len(recall)
    ndcg = sum(ndcg) / len(ndcg)

    return recall, ndcg


if __name__ == "__main__":    
    dataset = load_dataset("Cohere/miracl-en-queries-22-12", split="dev")
    dataset = dataset.to_pandas().to_dict(orient='records')

    ##############################
    ### EXPERIMENT 1: BASELINE ###
    print('EXPERIMENT 1: BASELINE')
    print("\n")

    # BM25
    searcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    recall, ndcg = run_search(searcher, dataset)
    print(f'BM25 Recall@100: {recall:.4f}')
    print(f'BM25 nDCG@10: {ndcg:.4f}')
    print("\n")

    # # mDPR
    # encoder = TctColBertQueryEncoder('castorini/mdpr-tied-pft-msmarco')
    # searcher = FaissSearcher.from_prebuilt_index(
    #     'miracl-v1.0-en-mdpr-tied-pft-msmarco',
    #     encoder
    # )
    # recall, ndcg = run_search(searcher, dataset)
    # print(f'mDPR Recall@100: {recall:.4f}')
    # print(f'mDPR nDCG@10: {ndcg:.4f}')
    # print("\n")

    # # Hybrid
    # ssearcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    # encoder = TctColBertQueryEncoder('castorini/mdpr-tied-pft-msmarco')
    # dsearcher = FaissSearcher.from_prebuilt_index(
    #     'miracl-v1.0-en-mdpr-tied-pft-msmarco',
    #     encoder
    # )
    # hsearcher = HybridSearcher(dsearcher, ssearcher)
    # recall, ndcg= run_search(hsearcher, dataset)
    # print(f'Hybrid Recall@100: {recall:.4f}')
    # print(f'Hybrid nDCG@10: {ndcg:.4f}')
    # print("\n")

    ########################################################
    ### EXPERIMENT 2: QUERY EXPANSION - ZERO SHOT PROMPT ###
    print('EXPERIMENT 2: QUERY EXPANSION - ZERO SHOT PROMPT')
    print("\n")

    # BM25
    searcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    data_expanded_zero_shot = get_query_expansion_dataset(dataset, option='zero-shot')
    #data_expanded_zero_shot_test = get_query_expansion_dataset(dataset[:10], option='zero-shot')

    recall, ndcg = run_search(searcher, data_expanded_zero_shot)
    print(f'BM25 Recall@100: {recall:.4f}')
    print(f'BM25 nDCG@10: {ndcg:.4f}')
    print("\n")

    #######################################################
    ### EXPERIMENT 3: QUERY EXPANSION - ONE SHOT PROMPT ###
    print('EXPERIMENT 3: QUERY EXPANSION - ONE SHOT PROMPT')
    print("\n")

    # BM25
    searcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    data_expanded_one_shot = get_query_expansion_dataset(dataset, option='one-shot')

    recall, ndcg = run_search(searcher, data_expanded_one_shot)
    print(f'BM25 Recall@100: {recall:.4f}')
    print(f'BM25 nDCG@10: {ndcg:.4f}')
    print("\n")

    #########################################################
    ### EXPERIMENT 4: QUERY EXPANSION - MULTI SHOT PROMPT ###
    print('EXPERIMENT 4: QUERY EXPANSION - MULTI SHOT PROMPT')
    print("\n")

    # BM25
    searcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    data_expanded_multi_shot = get_query_expansion_dataset(dataset, option='multi-shot')

    recall, ndcg = run_search(searcher, data_expanded_multi_shot)
    print(f'BM25 Recall@100: {recall:.4f}')
    print(f'BM25 nDCG@10: {ndcg:.4f}')
    print("\n")

    #####################################################
    ### EXPERIMENT 5: QUERY EXPANSION - ANSWER PROMPT ###
    print('EXPERIMENT 4: QUERY EXPANSION - ANSWER PROMPT')
    print("\n")

    # BM25
    searcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    data_expanded_answer = get_query_expansion_dataset(dataset, option='answer')

    recall, ndcg = run_search(searcher, data_expanded_answer)
    print(f'BM25 Recall@100: {recall:.4f}')
    print(f'BM25 nDCG@10: {ndcg:.4f}')
    print("\n")

    ##########################################################
    ### EXPERIMENT 6: QUERY EXPANSION - DIFFERENT LANGUAGE ###
    # use best working prompt technique for different languages
    # TODO

    ############################################################
    ### EXPERIMENT 7: QUERY EXPANSION - WITH CLASSIC METHODS ###
    # use methods such as synonym expansion, etc.
    # TODO