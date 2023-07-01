from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.chains.llm import LLMChain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
import numpy as np
from sklearn.cluster import KMeans
from langchain.vectorstores import Chroma, FAISS
from sklearn.preprocessing import StandardScaler
from prompts import Graph_Answer_Gen_Template
from langchain.prompts import PromptTemplate
from Graph_Queries import find_shortest_paths, query_inter_relationships 

def generate_answer(llm, relationships_list, source_list, target_list, inter_multi_hop_list, inter_direct_list, question, source, target, gene_string: Optional[str] = None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    multi_hop = ', '.join(relationships_list)
    source_sentences = ','.join(source_list)
    target_sentences = ','.join(target_list)
    inter_multi_hop_sentences = ','.join(inter_multi_hop_list)
    inter_direct_sentences = ','.join(inter_direct_list)
    sep_1 = f"Indirect relations between {source} and {target}:"
    sep2 = f"Direct relations from {source}:"
    sep3 = f"Direct relations from {target}:"
    sep4 = f"Indirect inter-relations between targets of {source} and {target}:"
    sep5 = f"Direct relations of the targets of {source} and {target}"
    if gene_string:
        sentences = '\n'.join([sep_1, multi_hop, sep2, source_sentences, sep3, target_sentences, sep4, inter_multi_hop_sentences, sep5, inter_direct_sentences, gene_string])
    else:
        sentences = '\n'.join([sep_1, multi_hop, sep2, source_sentences, sep3, target_sentences, sep4, inter_multi_hop_sentences, sep5, inter_direct_sentences])
    answer = gen_chain.run(input=sentences, question=question)
    print(answer)
    return answer

#########################################################################################################################################################################################

def cluster_and_select(sentences_list):
    hf = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True})

    sentences = sentences_list
    embeddings_array = np.array(hf.embed_documents(sentences))
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(embeddings_array)

    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    cluster_documents = {}
    for i, label in enumerate(cluster_labels):
        document = sentences[i]
        if label not in cluster_documents:
            cluster_documents[label] = document

    final_result = list(cluster_documents.values())
    print("done clustering")
    return final_result

def embed_and_select(sentences_list, question):
    hf = HuggingFaceEmbeddings(
    model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True})

    sentences = sentences_list
    db = Chroma.from_texts(sentences_list, hf)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)

    final_result = [doc.page_content for doc in docs]

    print("done embedding")
    return final_result

#####################################################################################################################################################################################################

class KnowledgeGraphRetrieval:
    def __init__(self, uri, username, password, llm, entity_types):
        self.graph = Graph(uri, auth=(username, password))
        self.llm = llm
        self.entity_types = entity_types  # Store the entity_types dictionary as an instance variable

    def _call(self, names_list, question, generate_an_answer, related_interactions):
        if related_interactions == True:
            result = find_shortest_paths(self.graph, "Test", names_list, self.entity_types, repeat=True)
        else:
            result = find_shortest_paths(self.graph, "Test", names_list, self.entity_types, repeat=True)

        # Check if result is a tuple of length 2
        if isinstance(result, tuple) and len(result) == 5:
            # Unpack result into relationship_context and associated_genes_string
            unique_relationships_list, unique_target_paths_list, unique_source_paths_list, final_path_nodes, gene_string = result
        else:
            # If not, relationship_context is result and associated_genes_string is an empty string
            unique_relationships_list, unique_target_paths_list, unique_source_paths_list, final_path_nodes = result
            gene_string = ""

        if len(unique_target_paths_list) > 15:
            print("number of unique target paths:")
            print(len(unique_target_paths_list))
            clustered_unique_target_paths_list = cluster_and_select(unique_target_paths_list)
            unique_target_paths_list = embed_and_select(clustered_unique_target_paths_list, question)
        else:
            unique_target_paths_list = unique_target_paths_list

        if len(unique_source_paths_list) > 15:
            clustered_unique_source_paths_list = cluster_and_select(unique_source_paths_list)
            unique_source_paths_list = embed_and_select(clustered_unique_source_paths_list, question)
            print("number of unique source paths:")
            print(len(unique_source_paths_list))        
        else:
            unique_source_paths_list = unique_source_paths_list

        if len(unique_relationships_list) >15:
            clustered_unique_relationships_list = cluster_and_select(unique_relationships_list)
            unique_relationships_list = embed_and_select(clustered_unique_relationships_list, question)
            print("number of unique relationships:")
            print(len(unique_relationships_list))
        else:
            unique_relationships_list = unique_relationships_list

        relationships_inter_list, relationships_direct_list, source_and_target_nodes  = query_inter_relationships(self.graph, final_path_nodes)
        
        if len(relationships_inter_list) > 15:
            print("number of unique inter_relationships:")
            print(len(relationships_inter_list))
            clustered_inter_relationships = cluster_and_select(relationships_inter_list)
            unique_relationships_inter_list = embed_and_select(clustered_inter_relationships, question)
        else:
            unique_relationships_inter_list = relationships_inter_list
            print("number of unique inter_relationships:")
            print(len(unique_relationships_inter_list))

        if len(relationships_direct_list) > 15:
            print("number of unique inter_direct_relationships:")
            print(len(relationships_direct_list))
            clustered_inter_direct_relationships = cluster_and_select(relationships_direct_list)
            unique_relationships_direct_list = embed_and_select(clustered_inter_direct_relationships, question)
        else:
            unique_relationships_direct_list = relationships_direct_list
            print("number of unique inter_relationships:")
            print(len(unique_relationships_direct_list))

        all_nodes = set()
        all_nodes.update(source_and_target_nodes)
        all_nodes.update(final_path_nodes)
        print("all nodes:")
        print(len(all_nodes))
        print(all_nodes)
        
########################################################################################################
        
        if generate_an_answer == True:
            final_context = generate_answer(llm=self.llm, 
                                            relationships_list=unique_relationships_list,
                                            question=question,
                                            source_list=unique_source_paths_list,
                                            target_list=unique_target_paths_list,
                                            inter_multi_hop_list=unique_relationships_inter_list,
                                            inter_direct_list=unique_relationships_direct_list,
                                            source=names_list[0],
                                            target=names_list[1]
                                            )
            answer = final_context


        response = {"result": answer, 
                    "multi_hop_relationships": unique_relationships_list,
                    "source_relationships": unique_source_paths_list,
                    "target_relationships": unique_target_paths_list,
                    "inter_multi_hop_relationships": unique_relationships_inter_list,
                    "inter_direct_relationships": unique_relationships_direct_list,
                    "all_nodes": all_nodes}

        if gene_string:
            print(gene_string)

            if generate_an_answer == True:
                del final_context
                final_context = generate_answer(llm=self.llm, 
                                                relationships_list=unique_relationships_list,
                                                question=question,
                                                source_list=unique_source_paths_list,
                                                target_list=unique_target_paths_list,
                                                inter_multi_hop_list=unique_relationships_inter_list,
                                                inter_direct_list=unique_relationships_direct_list,
                                                source=names_list[0],
                                                target=names_list[1],
                                                gene_string=gene_string
                                                )
                answer = final_context

        return response


