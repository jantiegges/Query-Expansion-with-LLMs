from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# Answer prompt
ANSWER_PROMPT = ChatPromptTemplate.from_template('Write a passage that answers the following query: {query}')

# Answer with pseudo-relevant feedback
ANSWER_PRF_PROMPT = ChatPromptTemplate.from_template('Write a passage that answers the given query based on the context: \
                                                      \n\nContext:{doc_1}\n{doc_2}\n{doc_3}\nQuery:{query}\nPassage:')

# Provide keywords
KEYWORDS_PROMPT = ChatPromptTemplate.from_template('Write a list of keywords for the following query: {query}')

# Provide keywords given pseudo-relevant feedback
KEYWORDS_PRF_PROMPT = ChatPromptTemplate.from_template('Write a list of keywords for the given query based on the context: \
                                                      \n\nContext:{doc_1}\n{doc_2}\n{doc_3}\nQuery:{query}\nKeywords:')

# Chain of thought prompt from Table 3 of Jagerman et. al
COT_PROMPT = ChatPromptTemplate.from_template('Answer the following query:\n {query}\nGive the rationale before answering')

# Chain of thought prompt with pseudo-relevant feedback from Jagerman et. al
COT_PRF_PROMPT = ChatPromptTemplate.from_template('Answer the following query based on the context:\nContext:{doc_1}\n \
                                                   {doc_2}\n{doc_3}\nQuery: {query}\nGive the rationale before answering')

# Chain of thought prompt but telling the model to give a short answer
COT_PROMPT_SHORT = ChatPromptTemplate.from_template('Answer the following question:\n {query} Give the rationale before answering. Keep your whole answer very short.')

# Chain of thought prompt for german
COT_PROMPT_SHORT_DE = ChatPromptTemplate.from_template('Beantworte die folgende Frage:\n {query} Erkläre vor dem antworten den Hintegrund. Fasse dich bei der Antwort sehr kurz.')

# Chain of thought prompt for spanish
COT_PROMPT_SHORT_ES = ChatPromptTemplate.from_template('Contesta la siguiente pregunta:：\n {query} Da la razón antes de responder. Mantenga toda su respuesta muy breve.')

# Chain of thought prompt for french
COT_PROMPT_SHORT_FR = ChatPromptTemplate.from_template('Répondez à la question suivante:\n {query} Donnez le raisonnement avant de répondre. La réponse doit être très courte.')

# Chain of thought prompt for chinese
COT_PROMPT_SHORT_ZH = ChatPromptTemplate.from_template('回答下列问题：\n {query} 回答前请说明理由。整个答案要非常简短。')

PROMPTS = {'answer': ANSWER_PROMPT,
           'answer-prf': ANSWER_PRF_PROMPT,
           'keywords': KEYWORDS_PROMPT,
           'keywords-prf': KEYWORDS_PRF_PROMPT,
           'chain-of-thought': COT_PROMPT,
           'chain-of-thought-prf': COT_PRF_PROMPT,
           'chain-of-thought-short': COT_PROMPT_SHORT,
           'chain-of-thought-de': COT_PROMPT_SHORT_DE,
           'chain-of-thought-es': COT_PROMPT_SHORT_ES,
           'chain-of-thought-fr': COT_PROMPT_SHORT_FR,
           'chain-of-thought-zh': COT_PROMPT_SHORT_ZH}