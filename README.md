# Multilingual Information Retrieval with Large Language Model-Driven Query Expansion
Information Retrieval (IR) is a critical task in web search and data mining.
Recent work has shown the potential of Large Language Models (LLMs) to expand search queries with relevant information, leading to improved IR.
However, LLM query expansion has only been considered for English IR tasks.
In this paper, we investigate LLM query expansion across five different languages: English, German, French, Spanish and simplified Chinese, and verify the effect of prompting observed in previous literature. Overall, we find LLM query expansion improves upon baseline retrieval methods in almost all cases, with short-Chain-of-thought prompting leading to the best results. Of the two language models considered (GPT 3.5 Turbo and Llama 2 7B), we found that GPT 3.5 Turbo consistently lead to better results, which may be due to the fact that it has more model parameters. We discuss several avenues for future work on multilingual IR.

Make sure to read our paper [here](paper.pdf).

The outputs from all of our experiments are shown in `experiments.ipynb`.

## Reproducing the results
1. Make sure to have miniconda/anaconda installed.
2. Follow these installation guidelines for `pyserini` (https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)
3. Install the required python packages with `pip install -r requirements.txt`
4. Create a `.env` file in the root directory and add the following variables:
    ```
    OPENAI_API_KEY=<your openai api key>
    COHERE_API_KEY=<your cohere api key>
    ANYSCALE_API_KEY=<your anyscale api key>
    ```
5. Make sure to have account balance on your OPEN AI/Cohere/Anyscale account. The codes will use the API to generate query expansions.
    - If you do not want to create new queries, our expanded queries are available for download [here](https://drive.google.com/file/d/1axe5vcsMYgsj8wougg0cRA9n1I_qNfF2/view?usp=sharing). To use these queries, unzip `expanded-queries.zip` inside the `data` folder.

⚠️ **Warning**: The code loads indexes that are big. Make sure to have around 60GB of free disk space.
