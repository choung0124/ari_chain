import psycopg2
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.pgvector import PGVector
import psycopg2
from transformers import BioGptTokenizer, BioGptForCausalLM
import torch

# model_name='pritamdeka/S-PubMedBert-MS-MARCO',
# ORDER BY hf_BERT_embedding_vector <-> %s::vector LIMIT 5;

# model_name='tavakolih/all-MiniLM-L6-v2-pubmed-full',
# ORDER BY all_mini_pubmed_vectors <-> %s::vector LIMIT 5;
### BEST SO FAR

# model_name='TimKond/S-BioLinkBert-MedQuAD',
# ORDER BY biolinkbert_vectors <-> %s::vector LIMIT 5;

# model_name='kamalkraj/BioSimCSE-BioLinkBERT-BASE',
# ORDER BY biosimcse_vectors <-> %s::vector LIMIT 5;

# model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
# ORDER BY pubmed_bert_vectors <-> %s::vector LIMIT 5;

# WHERE tsv @@ plainto_tsquery('english', %s)
# ORDER BY pubmed_bert_vectors <#> %s::vector LIMIT 5;
# """, (question, query_vector,))

import itertools
from sentence_transformers import CrossEncoder

query_embedding_model = HuggingFaceEmbeddings(
    model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def rerank(query, results):
    # re-rank
    encoder = CrossEncoder('pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb')
    scores = encoder.predict([(query, item[3]) for item in results])
    # sort results
    sorted_results = [(score, *result) for score, result in sorted(zip(scores, results), reverse=True)]
    # print all results
    for score, abstract_id, title, doi, abstract_text in sorted_results:
        print(score)
        print(abstract_id)
        print(title)
        print(doi)
        print(abstract_text)
        print()

    # return only the top 2 results
    return sorted_results[:2]

def hybrid_search(conn, query):
    query_vector = query_embedding_model.embed_query(query)
    with conn.cursor() as c:
        c.execute("""
            SELECT pmid, title, doi, abstract
            FROM pubmed_articles
            WHERE (pmid IS NOT NULL AND pmid <> '')
            AND (title IS NOT NULL AND title <> '')
            AND (doi IS NOT NULL AND doi <> '')
            AND (abstract IS NOT NULL AND abstract <> '')
            AND to_tsvector('english', title || ' ' || abstract) @@ plainto_tsquery('english', %s)
            ORDER BY pubmed_bert_vectors <#> %s::vector DESC, ts_rank_cd(to_tsvector('english', title || ' ' || abstract), plainto_tsquery('english', %s)) DESC
            LIMIT 1000
        """, (query, query_vector, query))
        results = c.fetchall()
        print(results)
    return results

query = "sildenafil interacts with PDE5A"

with psycopg2.connect(
    dbname='pubmed',
    user="hschoung",
    password="Reeds0124",
    host="localhost",
    port="5432"
) as conn:
    results = hybrid_search(conn, query)
    print(f"Results: {results}")  # Print results
    if results:
        results = rerank(query, results)
    else:
        print("No results found.")

print("sorted results:")
for score, abstract_id, title, doi, abstract_text in results:
    print(score)
    print(abstract_id)
    print(title)
    print(doi)
    print(abstract_text)
    print()
