# Improved Information Retrieval with Query Expansion
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
   
## Run Experiments

⚠️ **Warning**: The code loads indexes that are big. Make sure to have around 60GB of free disk space.

1. Run `experiments.ipynb` to re-run all experiments. 
