import psycopg2
from datasets import load_dataset
from tqdm import tqdm
import psycopg2.extras

def main():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="comp550-final",
        user="postgres",
        password="password"
    )
    cur = conn.cursor()

    # TODO: add vector & embedding columns

    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS corpusfr (
            docid TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            emb vector(768)
        );
    """)
    conn.commit()

    # stream the dataset bc it's too big to fit in memory
    dataset = load_dataset("Cohere/miracl-fr-corpus-22-12", split="train", streaming=True)
    batch_size = 1000  # Set the size of each insert batch
    batch = []
    count = 1
    for doc in tqdm(dataset, total=14600000):  # Assuming the total number of rows is 14.6 million
        batch.append((doc['docid'], doc['title'], doc['text'], doc['emb']))
        # TODO: compute mDPG embedding
        if len(batch) >= batch_size:
            execute_batch(cur, batch)
            #break  # Stop after the first batch
            count += 1
            batch = []
            if count == 10:
               break

    # TODO: add different index tables

    # TEST: Similarity vector search
    # get most similar row to first one
    cur.execute("SELECT * FROM corpusfr WHERE docid != '10066761#6' ORDER BY emb <-> (SELECT emb FROM corpusfr WHERE docid = '10066761#6') LIMIT 5;")

    # print the text of the most similar row
    rows = cur.fetchall()
    for row in rows:
        print(row[2])

    # TODO: add tests on computing metrics using the queries dataset

    # TODO: Implement different methods for information retrieval and ranking
    # METHODOLOGY: Preprocess Query -> Apply Retrieval Method -> Apply Re-Ranking Method -> Select Results

    # delete all rows
    cur.execute("DELETE FROM corpusfr;")
    conn.commit()

    cur.close()
    conn.close()

def execute_batch(cur, batch):
    insert_query = """
        INSERT INTO corpusfr (docid, title, text, emb)
        VALUES %s
        ON CONFLICT (docid) DO NOTHING;
    """
    
    psycopg2.extras.execute_values(cur, insert_query, batch)

if __name__ == "__main__":
    main()
