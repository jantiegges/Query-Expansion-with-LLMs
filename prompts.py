from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "Was there a year 0?", "output": "Existence of year 0 in calendars, astronomical year numbering, Gregorian and Julian calendars, year zero in ISO 8601, BC and AD year labeling, proleptic Gregorian calendar, historians' view on year 0, representation of year 1 BC in astronomical terms, usage of negative years in history"}
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_examples_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

one_example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, 
    examples=examples[:1]
)

# prompt for zero-shot query expansion
ZERO_SHOT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "As a query expansion AI, develop detailed yet succinct expanded queries. Focus on generating a set of relevant phrases and terms that thoroughly cover the key aspects of the given query. Your response should be comprehensive enough to capture a range of related concepts and synonyms, but also succinct to maintain focus."),
        ("human", "Query: {query}"),
        ("ai", "Detailed and succinct phrases and keywords: ")
    ]
)

# prompt for one-shot query expansion
ONE_SHOT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a query expansion AI bot. Given a query, provide a concise expanded query with essential keywords and phrases related to the original query, suitable for information retrieval systems like BM25. Answer with a maximum of 4 short sentences."),
        one_example_prompt,
        ("human", "Query: {query}"),
        ("ai", "Expanded phrases and keywords: ")
    ]
)

# prompt for multi-shot query expansion
MULTI_SHOT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a phrase-focused query expansion AI bot. Given a query, generate essential phrases and keywords that are semantically related to the original query, aimed at enhancing document retrieval in information retrieval systems like BM25. Focus on short, context-rich phrases that encapsulate key aspects of the query."),
        few_examples_prompt,
        ("human", "Query: {query}"),
        ("ai", "Expanded query: ")
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a question answering AI bot. Given a question, provide a short answer. Answer with a maximum of 4 short sentences."),
        ("human", "Question: {question}"),
        ("ai", "Answer: ")
    ]
)

# Q2D Zero-shot prompt from Table 3 of Jagerman et. al: https://arxiv.org/pdf/2305.03653.pdf
Q2D_ZS_PROMPT = ChatPromptTemplate.from_template('Write a passage that answers the following query: {query}')

# Q2D Zero-shot prompt with pseudo-relevant feedback from Jagerman et. al
Q2D_ZS_PRF_PROMPT = ChatPromptTemplate.from_template('Write a passage that answers the given query based on the context: \
                                                      \n\nContext:{doc_1}\n{doc_2}\n{doc_3}\nQuery:{query}\nPassage:')

# Q2E Zero-shot prompt from Jagerman et. al
Q2E_ZS_PROMPT = ChatPromptTemplate.from_template('Write a list of keywords for the following query: {query}')

# Q2E Zero-shot prompt with pseudo-relevant feedback from Jagerman et. al
Q2E_ZS_PRF_PROMPT = ChatPromptTemplate.from_template('Write a list of keywords for the given query based on the context: \
                                                      \n\nContext:{doc_1}\n{doc_2}\n{doc_3}\nQuery:{query}\nKeywords:')

# Chain of thought prompt from Table 3 of Jagerman et. al
COT_PROMPT = ChatPromptTemplate.from_template('Answer the following query:\n {query}\nGive the rational before answering')

# Chain of thought prompt with pseudo-relevant feedback from Jagerman et. al
COT_PRF_PROMPT = ChatPromptTemplate.from_template('Answer the following query based on the context:\nContext:{doc_1}\n \
                                                   {doc_2}\n{doc_3}\nQuery: {query}\nGive the rational before answering')

# Chain of thought prompt for different languages
COT_PROMPT_FR = ChatPromptTemplate.from_template('Répondez à la question suivante:\n {query}\nDonnez le rationnel avant de répondre.')

PROMPTS = {'q2d-zs': Q2D_ZS_PROMPT,
           'q2d-zs-prf': Q2D_ZS_PRF_PROMPT,
           'q2e-zs': Q2E_ZS_PROMPT,
           'q2e-zs-prf': Q2E_ZS_PRF_PROMPT,
           'chain-of-thought': COT_PROMPT,
           'chain-of-thought-prf': COT_PRF_PROMPT,
           'chain-of-thought-fr': COT_PROMPT_FR,
           'zero-shot': ZERO_SHOT_PROMPT,   # older prompts
           'one-shot': ONE_SHOT_PROMPT,
           'multi-shot': MULTI_SHOT_PROMPT,
           'answer': ANSWER_PROMPT}