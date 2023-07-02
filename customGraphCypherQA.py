from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import PubMedRetriever
from py2neo import Graph
import numpy as np
from sklearn.cluster import KMeans
from langchain.vectorstores import Chroma, FAISS
from sklearn.preprocessing import StandardScaler
from prompts import Coherent_sentences_template, Graph_Answer_Gen_Template, Graph_Answer_Gen_Template_airo
from langchain.prompts import PromptTemplate
from Custom_Agent import get_similar_compounds
from tqdm import tqdm

def get_source_and_target_paths(graph: Graph, label: str, names: List[str]) -> Tuple[List[Relationship], List[Relationship]]:
    query_source = f"""
    MATCH path=(source:{label})-[*1..2]->(node)
    WHERE toLower(source.name) = toLower("{names[0]}")
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    query_target = f"""
    MATCH path=(target:{label})-[*1..2]->(node)
    WHERE toLower(target.name) = toLower("{names[1]}")
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    print(query_source)
    print(query_target)
    source_results = list(graph.run(query_source))
    target_results = list(graph.run(query_target))
    source_paths = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in source_results]
    target_paths = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in target_results]
    source_relationships = [construct_relationship_string(record['path_nodes'], record['path_relationships']) for record in source_results]
    target_relationships = [construct_relationship_string(record['path_nodes'], record['path_relationships']) for record in target_results]
    print(source_relationships)
    all_path_nodes = set()
    for record in source_results:
        all_path_nodes.update(record['path_nodes'])
    for record in target_results:
        all_path_nodes.update(record['path_nodes'])
    print("source and target nodes:")
    print(len(all_path_nodes))
    #print(all_path_nodes)
    return source_paths, target_paths, source_relationships, target_relationships, all_path_nodes

def construct_path_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = [f"{nodes[i]} -> {relationships[i]} -> {nodes[i + 1]}" for i in range(len(nodes) - 1)]
    return " -> ".join(path_elements)

def construct_relationship_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = [f"{nodes[i]} -> {relationships[i]} -> {nodes[i + 1]}" for i in range(len(nodes) - 1)]
    return ", ".join(path_elements)

def find_shortest_paths(graph: Graph, label: str, names: List[str], entity_types: Dict[str, str], repeat: bool) -> List[Dict[str, Any]]:

    names_conditions = f'WHERE toLower(source.name) = toLower("{names[0]}") AND toLower(target.name) = toLower("{names[1]}")'
    query = f"""
    MATCH (source:{label}), (target:{label})
    {names_conditions}
    MATCH p = allShortestPaths((source)-[*]-(target))

    WITH p, [r IN relationships(p) WHERE type(r) = "ASSOCIATED_WITH" | startNode(r).name] AS associated_genes
    WITH p, associated_genes, [rel IN relationships(p) | type(rel)] AS path_relationships

    RETURN [node IN nodes(p) | node.name] AS path_nodes, associated_genes, path_relationships
    """
    print(query)
    result = graph.run(query)
    if not result and repeat==True:
        source_entity_type = entity_types[f"{names[0]}"]
        target_entity_type = entity_types[f"{names[1]}"]
        if source_entity_type == 'Drug':
            source_test_query = f"""
            MATCH (p:{label})
            WHERE p.name = "{names[0]}"
            RETURN p
            """
            source_test_result = graph.run(source_test_query)
            if not source_test_result:
                similar_compounds = get_similar_compounds({names[0]}, 20)
                for compound in similar_compounds[1:]:
                    source_similar_compounds_test_query = f"""
                    MATCH (p:{label})
                    WHERE p.name = "{compound}"
                    RETURN p
                    """
                    print(source_similar_compounds_test_query)           # start from index 1 to skip the first similar compound
                    source_similar_compounds_test_result = graph.run(source_similar_compounds_test_query)
                    if source_similar_compounds_test_result:
                        names[0] = compound
                        break

        if target_entity_type == 'Drug':
            target_test_query = f"""
            MATCH (p:{label})
            WHERE p.name = "{names[1]}"
            RETURN p
            """            
            target_test_result = graph.run(target_test_query)
            if not target_test_result:
                similar_compounds = get_similar_compounds({names[1]}, 20)
                for compound in similar_compounds[1:]:  # start from index 1 to skip the first similar compound
                    target_similar_compounds_test_query = f"""
                    MATCH (p:{label})
                    WHERE p.name = "{compound}"
                    RETURN p
                    """
                    print(target_similar_compounds_test_query)             # start from index 1 to skip the first similar compound
                    target_similar_compounds_test_result = graph.run(target_similar_compounds_test_query)
                    if target_similar_compounds_test_result:
                        names[1] = compound
                        break

    result = graph.run(query)
    #if not result and source_entity_type == "Drug":
        
    # Initialize a set to store unique associated genes
    associated_genes_set = set()
    unique_relationships = set()
    unique_source_paths = set()
    unique_target_paths = set()
    final_path_nodes = set()
    unique_graph_rels = set()
    for record in result:
        path_nodes = record['path_nodes']
        associated_genes_list = record['associated_genes']
        path_relationships = record['path_relationships']
        # Add the genes to the set
        final_path_nodes.update(record['path_nodes'])
        # Construct and add the relationship strings to the set
        path_string = construct_path_string(path_nodes, path_relationships)
        rel_string = construct_relationship_string(path_nodes, path_relationships)
        unique_graph_rels.add(rel_string)
        unique_relationships.add(path_string)
    
    source_paths, target_paths, source_relationships, target_relationships, source_and_target_nodes = get_source_and_target_paths(graph, label, names)

    for path in source_paths:
        unique_source_paths.add(path)
    # Construct and add the target path relationship strings to the set
    for path in target_paths:
        unique_target_paths.add(path)

    for node in source_and_target_nodes:
        final_path_nodes.add(node)

    for rel in source_relationships:
        unique_graph_rels.add(rel)

    for rel in target_relationships:
        unique_graph_rels.add(rel)

    print("number of nodes:")
    print(len(final_path_nodes))
    #print(final_path_nodes)
    # Remove the source and target node names from the associated genes set
    lower_names = {name.lower() for name in names}
    associated_genes_set = {gene for gene in associated_genes_set if gene.lower() not in lower_names}

    # Convert unique_relationships set to list
    unique_relationships_list = list(unique_relationships)
    unique_source_paths_list = list(unique_source_paths)
    unique_target_paths_list = list(unique_target_paths)
    unique_graph_rels_list = list(unique_graph_rels)
    #print(unique_graph_rels)
    # Check if there are associated genes and return accordingly
    if associated_genes_set:
        associated_genes_list = list(associated_genes_set)
        gene_string = f"The following genes are associated with both {names[0]} and {names[1]}: {', '.join(associated_genes_list)}"
        print(gene_string)
        return unique_relationships_list, unique_target_paths_list, unique_source_paths_list, unique_graph_rels_list, final_path_nodes, gene_string
    else:
        print("There are no associated genes.")
        #print(unique_relationships_list)

        return unique_relationships_list, unique_target_paths_list, unique_source_paths_list, unique_graph_rels_list, final_path_nodes
    
def query_inter_relationships(graph: Graph, nodes:List[str]) -> str:
    query_parameters = {"nodes": list(nodes)}

    # Query for direct relationships to and from nodes
    direct_relationships_query = """
    UNWIND $nodes AS nodeName
    MATCH (n:Test) WHERE n.name = nodeName
    WITH n
    LIMIT 10
    MATCH p = (n)-[*1..]-(m)
    WITH p, [rel IN relationships(p) WHERE startNode(rel) <> n | type(rel)] AS path_relationships
    RETURN DISTINCT [node IN nodes(p) | node.name] AS path_nodes, path_relationships
    """
    result_direct = graph.run(direct_relationships_query, **query_parameters)
    print(direct_relationships_query)

    inter_between_direct_query = """
    UNWIND $nodes AS nodeName
    MATCH (n:Test) WHERE n.name = nodeName
    MATCH (n)-[r]-(m)
    RETURN [n.name, m.name] AS path_nodes, [type(r)] AS path_relationships
    """
    result_inter_direct = graph.run(inter_between_direct_query, **query_parameters)
    print(inter_between_direct_query)

    # Combine results
    relationships_inter_direct = set()
    relationships_direct = set()
    all_nodes = set()
    graph_strings = set()

    for record in result_direct:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_path_string(path_nodes, path_relationships)
        graph_string = construct_relationship_string(path_nodes, path_relationships)
        graph_strings.add(graph_string)
        relationships_direct.add(rel_string)
        all_nodes.update(record['path_nodes'])
    for record in result_inter_direct:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_path_string(path_nodes, path_relationships)
        graph_string = construct_relationship_string(path_nodes, path_relationships)
        graph_strings.add(graph_string)
        relationships_inter_direct.add(rel_string)
        all_nodes.update(record['path_nodes'])

    relationships_direct_list = list(relationships_direct) if relationships_direct else []
    print("relationships_direct_list")
    print(relationships_direct_list)
    relationships_inter_direct_list = list(relationships_inter_direct) if relationships_inter_direct else []
    print("relationships_inter_direct_list")
    print(relationships_inter_direct_list)
    graph_strings_list = list(graph_strings)
    return relationships_direct_list, relationships_inter_direct_list, graph_strings_list, all_nodes

#######################################################################################################################################################################################

def generate_answer(llm, relationships_list, source_list, target_list, inter_direct_list, inter_direct_inter, question, source, target, gene_string: Optional[str] = None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template, input_variables=["input", "question"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    multi_hop = ', '.join(relationships_list)
    source_sentences = ','.join(source_list)
    target_sentences = ','.join(target_list)
    Inter_relationships = inter_direct_list + inter_direct_inter
    Inter_sentences = ','.join(Inter_relationships)
    sep_1 = f"Indirect relations between {source} and {target}:"
    sep2 = f"Direct relations from {source}:"
    sep3 = f"Direct relations from {target}:"
    sep4 = f"Relations between the targets of {source} and {target}"
    if gene_string:
        sentences = '\n'.join([sep_1, multi_hop, sep2, source_sentences, sep3, target_sentences, sep4, Inter_sentences, gene_string])
    else:
        sentences = '\n'.join([sep_1, multi_hop, sep2, source_sentences, sep3, target_sentences, sep4, Inter_sentences])
    answer = gen_chain.run(input=sentences, question=question)
    print(answer)
    return answer

def generate_answer_airo(llm, relationships_list, source_list, target_list, inter_direct_list, inter_direct_inter, question, source, target, gene_string: Optional[str] = None):
    prompt = PromptTemplate(template=Graph_Answer_Gen_Template_airo, input_variables=["question", 
                                                                                      "source", 
                                                                                      "target", 
                                                                                      "multihop_relations", 
                                                                                      "direct_relations_source",
                                                                                      "direct_relations_target",
                                                                                      "inter_relations"])
    gen_chain = LLMChain(llm=llm, prompt=prompt)
    multi_hop = ', '.join(relationships_list)
    source_sentences = ','.join(source_list)
    target_sentences = ','.join(target_list)
    Inter_relationships = inter_direct_list + inter_direct_inter
    Inter_sentences = ','.join(Inter_relationships)

    answer = gen_chain.run(question=question,
                           source=source,
                           target=target,
                           multihop_relations=multi_hop,
                           direct_relations_source=source_sentences,
                           direct_relations_target=target_sentences,
                           inter_relations=Inter_sentences)
    print(answer)
    return answer

#########################################################################################################################################################################################

def cluster_and_select(sentences_list, progress_callback=None):
    batch_size = 100
    total_iterations = len(sentences_list) // batch_size + 1

    embeddings_list = []
    for i in range(0, len(sentences_list), batch_size):
        batch_sentences = sentences_list[i:i+batch_size]

        # Construct HuggingFaceEmbeddings instance for the batch
        batch_hf = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True})

        # Embed documents for the batch
        batch_embeddings_array = np.array(batch_hf.embed_documents(batch_sentences))
        embeddings_list.append(batch_embeddings_array)

        # Update the progress bar
        if progress_callback:
            progress_callback((i + len(batch_sentences)) / len(sentences_list))

    # Concatenate embeddings from all batches
    embeddings_array = np.concatenate(embeddings_list)

    # Continue with the remaining code
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(embeddings_array)

    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    cluster_documents = {}
    for i, label in enumerate(cluster_labels):
        document = sentences_list[i]
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

    def _call(self, names_list, question, generate_an_answer, related_interactions, progress_callback=None):
        if related_interactions == True:
            result = find_shortest_paths(self.graph, "Test", names_list, self.entity_types, repeat=True)
        else:
            result = find_shortest_paths(self.graph, "Test", names_list, self.entity_types, repeat=True)

        # Check if result is a tuple of length 2
        if isinstance(result, tuple) and len(result) == 6:
            # Unpack result into relationship_context and associated_genes_string
            unique_relationships_list, unique_target_paths_list, unique_source_paths_list, unique_graph_rels, final_path_nodes, gene_string = result
        else:
            # If not, relationship_context is result and associated_genes_string is an empty string
            unique_relationships_list, unique_target_paths_list, unique_source_paths_list, unique_graph_rels, final_path_nodes,  = result
            gene_string = ""

        if len(unique_target_paths_list) > 15:
            print("number of unique target paths:")
            print(len(unique_target_paths_list))
            clustered_unique_target_paths_list = cluster_and_select(unique_target_paths_list, progress_callback)
            unique_target_paths_list = embed_and_select(clustered_unique_target_paths_list, question)
        else:
            unique_target_paths_list = unique_target_paths_list

        if len(unique_source_paths_list) > 15:
            print("number of unique source paths:")
            print(len(unique_source_paths_list))        
            clustered_unique_source_paths_list = cluster_and_select(unique_source_paths_list, progress_callback)
            unique_source_paths_list = embed_and_select(clustered_unique_source_paths_list, question)
        else:
            unique_source_paths_list = unique_source_paths_list

        if len(unique_relationships_list) >15:
            print("number of unique relationships:")
            print(len(unique_relationships_list))
            clustered_unique_relationships_list = cluster_and_select(unique_relationships_list, progress_callback)
            unique_relationships_list = embed_and_select(clustered_unique_relationships_list, question)
        else:
            unique_relationships_list = unique_relationships_list

        relationships_direct_list, relationships_inter_direct_list, inter_unique_graph_rels, source_and_target_nodes  = query_inter_relationships(self.graph, final_path_nodes)
        
        if len(relationships_direct_list) > 15:
            print("number of unique inter_direct_relationships:")
            print(len(relationships_direct_list))
            clustered_inter_direct_relationships = cluster_and_select(relationships_direct_list, progress_callback)
            unique_relationships_direct_list = embed_and_select(clustered_inter_direct_relationships, question)
        else:
            unique_relationships_direct_list = relationships_direct_list
            print("number of unique inter_direct_relationships:")
            print(len(unique_relationships_direct_list))

        if len(relationships_inter_direct_list) > 15:
            print("number of unique inter_direct_inter_relationships:")
            print(len(relationships_inter_direct_list))
            clustered_relationships_inter_direct_list = cluster_and_select(relationships_inter_direct_list, progress_callback)
            relationships_inter_direct_list = embed_and_select(clustered_relationships_inter_direct_list, question)
        else:
            relationships_inter_direct_list = relationships_inter_direct_list
            print("number of unique inter_direct_inter_relationships:")
            print(len(relationships_inter_direct_list))

        all_nodes = set()
        all_nodes.update(source_and_target_nodes)
        all_nodes.update(final_path_nodes)
        print("all nodes:")
        print(len(all_nodes))
        print(all_nodes)
        all_unique_graph_rels = set()
        all_unique_graph_rels.update(unique_graph_rels)
        all_unique_graph_rels.update(inter_unique_graph_rels)

########################################################################################################
        
        if generate_an_answer == True:
            final_context = generate_answer(llm=self.llm, 
                                            relationships_list=unique_relationships_list,
                                            question=question,
                                            source_list=unique_source_paths_list,
                                            target_list=unique_target_paths_list,
                                            inter_direct_list=unique_relationships_direct_list,
                                            inter_direct_inter=relationships_inter_direct_list,
                                            source=names_list[0],
                                            target=names_list[1]
                                            )
            #final_context = generate_answer_airo(llm=self.llm, 
            #                    relationships_list=unique_relationships_list,
            #                    question=question,
            #                    source_list=unique_source_paths_list,
            #                    target_list=unique_target_paths_list,
            #                    inter_direct_list=unique_relationships_direct_list,
            #                    inter_direct_inter=relationships_inter_direct_list,
            #                    source=names_list[0],
            #                    target=names_list[1]
            #                    )
            answer = final_context


        response = {"result": answer, 
                    "multi_hop_relationships": unique_relationships_list,
                    "source_relationships": unique_source_paths_list,
                    "target_relationships": unique_target_paths_list,
                    "inter_direct_relationships": unique_relationships_direct_list,
                    "inter_direct_inter_relationships": relationships_inter_direct_list,
                    "all_nodes": all_nodes,
                    "all_rels": all_unique_graph_rels}

        if gene_string:
            print(gene_string)

            if generate_an_answer == True:
                del final_context
                final_context = generate_answer(llm=self.llm, 
                                                relationships_list=unique_relationships_list,
                                                question=question,
                                                source_list=unique_source_paths_list,
                                                target_list=unique_target_paths_list,
                                                inter_direct_list=unique_relationships_direct_list,
                                                inter_direct_inter=relationships_inter_direct_list,
                                                source=names_list[0],
                                                target=names_list[1],
                                                gene_string=gene_string
                                                )
                #final_context = generate_answer_airo(llm=self.llm, 
                #                    relationships_list=unique_relationships_list,
                #                    question=question,
                #                    source_list=unique_source_paths_list,
                #                    target_list=unique_target_paths_list,
                #                    inter_direct_list=unique_relationships_direct_list,
                #                    inter_direct_inter=relationships_inter_direct_list,
                #                    source=names_list[0],
                #                    target=names_list[1]
                #                    )

                answer = final_context

        return response


