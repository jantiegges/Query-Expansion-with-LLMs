from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
import pickle
import copy
from prompts import PROMPTS
from utils.metrics import get_recall_at_100, get_nDCG_at_10

# set up LLM
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

def get_prf_docs(dataset, lang='en', k=3):
    """Find the top :k: most relevant documents using BM25 search with the original query"""

    # TODO: Add options for different languages
    if lang == 'en':
        searcher = LuceneSearcher.from_prebuilt_index('miracl-v1.0-en')
    else:
        raise ValueError(f'Invalid language: {lang}')

    prf_docs = {}
    for data in tqdm(dataset, total=len(dataset), desc='Searching'):
        query = data['query']
        results = searcher.search(query, k=k)
        prf_docs[query] = tuple([r.raw.partition('"text" : "')[-1][:-1] for r in results])

    return prf_docs


def get_query_expansion_dataset(dataset, chat, lang='en', option='q2d-zs', n_query_repeats=5):
    """Expands queries with names based on Table 3 of Jagerman et. al: https://arxiv.org/pdf/2305.03653.pdf. 
    As in Jagerman et. al, the original query is repeated :n_query_repeats: times in the expanded query."""
    # try to load the expanded dataset
    try:
        with open(f'./data/expanded-queries/{option}/miracl-en-queries-22-12-expanded-{option}-{n_query_repeats}-query-repeats.pkl','rb') as f:
            expanded_dataset = pickle.load(f)
        return expanded_dataset
    except:
        expanded_dataset = copy.deepcopy(dataset)
        if option[-3:] == 'prf':
            print('Getting prf documents...')
            prf_docs = get_prf_docs(dataset, lang=lang)
            print('Done.')

        assert option in PROMPTS.keys(), f'Prompt option not found: {option}.'

        for i in tqdm(range(len(dataset)), total=len(dataset), desc='Expanding queries'):
            query = dataset[i]['query']

            if option[-3:] == 'prf':
                doc_1, doc_2, doc_3 = prf_docs[query]
                messages = PROMPTS[option].format_messages(doc_1=doc_1, doc_2=doc_2, doc_3=doc_3, query=query)
            else:
                messages = PROMPTS[option].format_messages(query=query)

            # Repeat original query and combine with LLM results (see Equation 1 of Jagerman et. al)
            repeated_query = ' '.join([query for _ in range(n_query_repeats)])
            expanded_query = chat(messages).content
            expanded_dataset[i]['query'] = repeated_query + ' ' + expanded_query
            # print(expanded_dataset[i]['query'])
            # exit()

        with open(f'./data/expanded-queries/{option}/miracl-en-queries-22-12-expanded-{option}-{n_query_repeats}-query-repeats.pkl', 'wb') as f:
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
    print('EXPERIMENT 5: QUERY EXPANSION - ANSWER PROMPT')
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