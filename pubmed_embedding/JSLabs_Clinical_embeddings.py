from pyspark.sql import SparkSession
from sparknlp.base import *
from sparknlp.annotator import *
import sparknlp
import psycopg2

spark = sparknlp.start()

from tqdm import tqdm

conn = psycopg2.connect(
    dbname='pubmed',
    user="hschoung",
    password="Reeds0124",
    host="localhost",
    port="5432"
    )


def get_clinical_embeddings_pipeline():
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical_large", "en", "clinical/models") \
        .setInputCols(["document", "token"]) \
        .setOutputCol("word_embeddings")

    pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings])
    return pipeline

def fetch_all_abstracts(conn):
    with conn.cursor() as c:
        c.execute("""
            SELECT id, title, abstract
            FROM abstracts
            WHERE jslabs_clinical_embedding IS NULL;""")
        rows = c.fetchall()
        data = [(row[0], row[1] + ' ' + row[2]) for row in rows]
        return spark.createDataFrame(data, schema=["id", "text"])

def update_jslabs_clinical_embedding(conn, result):
    for row in tqdm(result.collect(), desc="Updating embeddings"):
        id, embeddings = row['id'], row['word_embeddings']
        embeddings_list = [e.embeddings.tolist() for e in embeddings]
        with conn.cursor() as c:
            c.execute("""
                UPDATE abstracts
                SET jslabs_clinical_embedding = %s
                WHERE id = %s;""",
                (embeddings_list, id))
            conn.commit()
        print(f"Embedded abstract {id}: {row['text']}")

def create_jslabs_clinical_embedding_column(conn):
    with conn.cursor() as c:
        c.execute("""
            ALTER TABLE abstracts
            ADD COLUMN IF NOT EXISTS jslabs_clinical_embedding double precision[];""")
        conn.commit()

def main(conn):
    create_jslabs_clinical_embedding_column(conn)
    abstracts_df = fetch_all_abstracts(conn)
    pipeline = get_clinical_embeddings_pipeline()
    pipeline_model = pipeline.fit(abstracts_df)
    result = pipeline_model.transform(abstracts_df)
    update_jslabs_clinical_embedding(conn, result)


