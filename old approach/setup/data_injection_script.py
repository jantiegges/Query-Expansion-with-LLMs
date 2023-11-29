import logging
from datasets import load_dataset
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm
from time import sleep

logging.basicConfig(filename='data_insertion.log', level=logging.INFO)

def setup_database(cur):
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS corpusfr (
            docid TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            emb vector(768)
        );
    """)

def insert_dataset(cur):
    batch_size = 10000
    last_processed_id = None

    while True:
        try:
            dataset = load_dataset("Cohere/miracl-fr-corpus-22-12", split="train", streaming=True)
            batch = []
            for doc in tqdm(dataset, total=14636953):
                batch.append((doc['docid'], doc['title'], doc['text'], doc['emb']))
                if len(batch) >= batch_size:
                    execute_batch_operation(cur, batch)
                    last_processed_id = batch[-1][0]  
                    batch = []
                    logging.info(f"Processed batch up to ID: {last_processed_id}")
            if batch:
                execute_batch_operation(cur, batch)
            break  
        except Exception as e:
            logging.error(f"Error loading or processing dataset: {e}, last processed ID was: {last_processed_id}")
            sleep(10) 

def execute_batch_operation(cur, batch):
    try:
        cur.execute("BEGIN;")
        query = "INSERT INTO corpusfr (docid, title, text, emb) VALUES (%s, %s, %s, %s)"
        execute_batch(cur, query, batch)
        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        logging.error(f"Error executing batch: {e}")

def create_bm25_index(cur):
    cur.execute("BEGIN;")
    cur.execute("DROP INDEX IF EXISTS idx_corpusfr;")
    cur.execute("""
        CREATE INDEX idx_corpusfr
        ON corpusfr
        USING bm25 ((corpusfr.*))
        WITH (
        text_fields='{"text": {"tokenizer": {"type": "default"}}}'
        );
    """)
    cur.execute("COMMIT;")

def create_hnsw_index(cur):
    cur.execute("""
        CREATE INDEX ON corpusfr 
        USING hnsw (emb vector_cosine_ops);
    """)

if __name__ == "__main__":
    connection_parameters = {
        'host': "localhost",
        'port': 5432,
        'dbname': "comp550-final",
        'user': "postgres",
        'password': "password"
    }

    with psycopg2.connect(**connection_parameters) as conn:
        with conn.cursor() as cur:
            setup_database(cur)
            insert_dataset(cur)

            print("Creating BM25 index...")
            create_bm25_index(cur)
            print("Creating HNSW index...")
            create_hnsw_index(cur)

            cur.close()