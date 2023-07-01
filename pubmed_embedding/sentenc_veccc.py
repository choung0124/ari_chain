import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import psycopg2
import numpy as np
import pickle

# Load the BioBERT model
model_name = 'dmis-lab/biobert-base-cased-v1.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_embedding(tokenized_abstract):
    # Split the tokenized abstract into chunks of 512 tokens or less
    tokens = tokenized_abstract.split()
    chunks = [tokens[i:i+512] for i in range(0, len(tokens), 512)]
    
    all_embeddings = []
    
    for chunk in chunks:
        # Convert chunk of tokens back to text
        chunk_text = ' '.join(chunk)

        # Tokenize the chunk
        inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        inputs.to(device)

        # Generate the embedding
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the embeddings of the [CLS] token (the first token), which represents the abstract-level embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    
    # Average the embeddings for each chunk to get a single embedding for the abstract
    embeddings = np.mean(all_embeddings, axis=0)

    return embeddings

def main():
    conn = psycopg2.connect(
        dbname="pubmed",
        user="hschoung",
        password="Reeds0124",
        host="localhost",
        port="5432"
    )
    c = conn.cursor()

    # Fetch all abstracts
    c.execute('SELECT id, processed_abstract FROM abstracts')
    abstracts = c.fetchall()

    # Generate and store embeddings
    for id, processed_abstract in tqdm(abstracts, desc="Generating embeddings"):
        embeddings = generate_embedding(processed_abstract)

        # Convert numpy array to bytes
        embeddings_bytes = pickle.dumps(embeddings)

        # Store the embeddings in the database
        c.execute('UPDATE abstracts SET embedding = %s WHERE id = %s', (embeddings_bytes, id))
        conn.commit()

    conn.close()

if __name__ == "__main__":
    main()

