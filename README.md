# Improved Information Retrieval with Query Expansion

## Setup
1. Make sure to have miniconda/anaconda installed.
2. Follow these installation guidelines (https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)
3. Install the required python packages with `pip install -r requirements.txt`
4. Create a `.env` file in the root directory and add the following variables:
    ```
    OPENAI_API_KEY=<your openai api key>
    COHERE_API_KEY=<your cohere api key>
    ANYSCALE_API_KEY=<your anyscale api key>
    ```
5. Make sure to have account balance on your accounts. The codes will use the API to generate query expansions.
   
## Run Experiments

⚠️ **Warning**: The code loads indexes that are big. Make sure to have around 30GB of free disk space.

1. Run `python main.py` to run all experiments. 