import os
import pubmed_parser as pp
import psycopg2
from tqdm import tqdm
import multiprocessing
import threading

def create_table_if_not_exists():
    conn = psycopg2.connect("dbname=pubmed user=hschoung password=Reeds0124 host=localhost")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid VARCHAR PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            doi VARCHAR,
            pmc VARCHAR
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_article(pmid, title, abstract, doi, pmc):
    conn = psycopg2.connect("dbname=pubmed user=hschoung password=Reeds0124 host=localhost")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO pubmed_articles (pmid, title, abstract, doi, pmc)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (pmid) DO NOTHING;
    """, (pmid, title, abstract, doi, pmc))
    conn.commit()
    cur.close()
    conn.close()

def process_file_wrapper(file_name):
    return process_file(file_name, "ftp.ncbi.nlm.nih.gov/pubmed/baseline")

def parse_pubmed_file(file_path):
    pubmed_data = pp.parse_medline_xml(file_path)

    for article in pubmed_data:
        pmid = article['pmid']
        title = article['title']
        abstract = article['abstract']
        doi = article['doi']
        pmc = article['pmc']
        if not abstract.strip():  # Check if the abstract is empty
            continue  # Skip to the next article if the abstract is empty
        insert_article(pmid, title, abstract, doi, pmc)

def process_file(file_name, directory_path):
    file_path = os.path.join(directory_path, file_name)
    if file_path.endswith('.xml.gz') and os.path.isfile(file_path):
        parse_pubmed_file(file_path)
        return True
    return False

def parse_all_files_in_directory(directory_path):
    files = [f for f in os.listdir(directory_path) if f.endswith('.xml.gz') and os.path.isfile(os.path.join(directory_path, f))]
    create_table_if_not_exists()

    with multiprocessing.Pool(processes=16) as pool:
        results = pool.imap_unordered(process_file_wrapper, files)
        for i, success in enumerate(tqdm(results, total=len(files), desc="Parsing files"), start=1):
            pass

# Replace 'path/to/your/directory' with the actual path to the directory containing the XML.gz files
parse_all_files_in_directory('ftp.ncbi.nlm.nih.gov/pubmed/baseline')
