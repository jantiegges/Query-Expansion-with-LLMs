# Improved Information Retrieval with Query Expansion

## Setup
1. Make sure to have miniconda/anaconda installed.
2. Follow these installation guidelines (https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)
3. Install the required python packages with `pip install -r requirements.txt`
4. Create a `.env` file in the root directory and add the following variables:
    ```
    OPENAI_API_KEY=<your openai api key>
    ```
5. Make sure to have account balance on your OpenAI account. The code will use the API to generate query expansions.
   
## Run Experiments

⚠️ **Warning**: The code loads indexes that are huge. Make sure to have around 120GB of free disk space.

1. Run `python main.py` to run all experiments. 