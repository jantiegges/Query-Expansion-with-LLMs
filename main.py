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
from utils.helpers import get_language_from_abbreviation

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


def get_query_expansion_dataset(dataset, chat, chat_name, lang='en', prompt='answer', n_query_repeats=5, verbose=False):
    """Expands queries with using one of the prompt templates from prompts.py. 
    As in Jagerman et. al, the original query is repeated :n_query_repeats: times in the expanded query."""
    # try to load the expanded dataset
    try:
        with open(f'./data/expanded-queries/{prompt}/{chat_name}-miracl-{lang}-queries-22-12-expanded-{prompt}-{n_query_repeats}-query-repeats.pkl','rb') as f:
            expanded_dataset = pickle.load(f)
        return expanded_dataset
    except:
        expanded_dataset = copy.deepcopy(dataset)
        lang_full = get_language_from_abbreviation(lang)
        if prompt[-3:] == 'prf':
            print('Getting prf documents...')
            prf_docs = get_prf_docs(dataset, lang=lang)
            print('Done.')

        assert prompt in PROMPTS.keys(), f'Prompt option not found: {prompt}.'

        for i in tqdm(range(len(dataset)), total=len(dataset), desc='Expanding queries'):
            query = dataset[i]['query']

            if prompt[-3:] == 'prf':
                doc_1, doc_2, doc_3 = prf_docs[query]
                messages = PROMPTS[prompt].format_messages(doc_1=doc_1, doc_2=doc_2, doc_3=doc_3, query=query)
            else:
                messages = PROMPTS[prompt].format_messages(query=query)

            # Repeat original query and combine with LLM results (see Equation 1 of Jagerman et. al)
            repeated_query = ' '.join([query for _ in range(n_query_repeats)])
            expanded_query = chat(messages).content
            expanded_dataset[i]['query'] = repeated_query + ' ' + expanded_query
            if verbose:
                print(expanded_dataset[i]['query'])
            # exit()

        with open(f'./data/expanded-queries/{prompt}/{chat_name}-miracl-{lang}-queries-22-12-expanded-{prompt}-{n_query_repeats}-query-repeats.pkl', 'wb') as f:
            pickle.dump(expanded_dataset, f)

        return expanded_dataset


def run_search(searcher, dataset, k=100):
    """Performs information retrieval using the :searcher: on the :dataset:.
      Finds the top :k: most relevant queries. Returns the recall and nDCG scores for the search.
    """
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
