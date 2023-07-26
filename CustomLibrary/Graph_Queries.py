from __future__ import annotations
from typing import Any, Dict, List, Optional
from py2neo import Graph
from CustomLibrary.Utils import get_similar_compounds
from typing import Tuple, Set

def get_node_label(graph: Graph, node_name: str) -> str:
    query = f"""
    MATCH (node)
    WHERE toLower(node.name) = toLower("{node_name}")
    RETURN head(labels(node)) AS FirstLabel, node.name AS NodeName
    """
    result = graph.run(query).data()
    if result:
        print(query)
        return result[0]['FirstLabel'], result[0]['NodeName']
    else:
        return None, None
    
def get_node_labels(graph: Graph, node_names: List[str]) -> Dict[str, str]:
    query = """
    UNWIND $node_names AS node_name
    MATCH (node)
    WHERE toLower(node.name) = toLower(node_name)
    RETURN node.name AS NodeName, head(labels(node)) AS FirstLabel
    """
    results = graph.run(query, node_names=node_names).data()
    return {result['NodeName']: result['FirstLabel'] for result in results}


def get_source_and_target_paths(graph: Graph, names: List[str]) -> Tuple[List[Relationship], List[Relationship]]:
    source_label, source_name = get_node_label(graph, names[0])
    target_label, target_name = get_node_label(graph, names[1])

    query_source = f"""
    MATCH path=(source:{source_label})-[*1..2]->(node)
    WHERE source.name = "{source_name}"
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """

    query_target = f"""
    MATCH path=(target:{target_label})-[*1..2]->(node)
    WHERE target.name = "{target_name}"
    WITH relationships(path) AS rels, nodes(path) AS nodes
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """

    print(query_source)
    print(query_target)
    source_results = list(graph.run(query_source))
    target_results = list(graph.run(query_target))
    source_paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in source_results]
    target_paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in target_results]
    print("source paths:")
    print(len(source_paths))
    print("target paths")
    print(len(target_paths))
    source_relationships = [construct_relationship_string(record['path_nodes'], record['path_relationships']) for record in source_results]
    target_relationships = [construct_relationship_string(record['path_nodes'], record['path_relationships']) for record in target_results]
    
    with open("sample.txt", "w") as file:
        source_rels_to_write = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in source_results]
        for string in source_rels_to_write:
            file.write(string + '\n')
        target_rels_to_write = [construct_path_string(record['path_nodes'], record['path_relationships']) for record in target_results]
        for string in target_rels_to_write:
            file.write(string + '\n')
    
    return source_paths, target_paths, source_relationships, target_relationships

def construct_path_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = []
    for node, relationship in zip(nodes, relationships):
        if node is None or relationship is None:
            continue  # Skip this element if the node or the relationship is None
        path_elements.append(f"{node} -> {relationship}")
    if nodes[-1] is not None:
        path_elements.append(nodes[-1])  # add the last node
    return " -> ".join(path_elements)

def construct_relationship_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = []
    for i in range(len(nodes) - 1):
        if nodes[i] is None or relationships[i] is None or nodes[i + 1] is None:
            continue  # Skip this element if any of the nodes or the relationship is None
        path_elements.append(f"{nodes[i]} -> {relationships[i]} -> {nodes[i + 1]}")
    return ", ".join(path_elements)

def find_shortest_paths(graph: Graph, names: List[str], entity_types: Dict[str, str], repeat: bool) -> List[Dict[str, Any]]:
    source_label, source_name = get_node_label(graph, names[0])
    target_label, target_name = get_node_label(graph, names[1])
    names_conditions = f'WHERE source.name = "{source_name}" AND target.name = "{target_name}"'
    query = f"""
    MATCH (source:{source_label }), (target:{target_label})
    {names_conditions}
    MATCH p = allShortestPaths((source)-[*]-(target))
    WITH p, [rel IN relationships(p) | type(rel)] AS path_relationships
    WITH relationships(p) AS rels, nodes(p) AS nodes, path_relationships
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """

    print(query)
    result = graph.run(query)

    if not result:
        source_entity_type = entity_types[f"{names[0]}"]
        target_entity_type = entity_types[f"{names[1]}"]
        if source_entity_type == 'Drug':
            source_test_query = f"""
            MATCH (p:Drug)
            WHERE p.name = "{names[0]}"
            RETURN p
            """
            source_test_result = graph.run(source_test_query)
            if not source_test_result:
                similar_compounds = get_similar_compounds({names[0]}, 20)
                for compound in similar_compounds[1:]:
                    source_similar_compounds_test_query = f"""
                    MATCH (p:Drug)
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
            MATCH (p:Drug)
            WHERE p.name = "{names[1]}"
            RETURN p
            """            
            target_test_result = graph.run(target_test_query)
            if not target_test_result:
                similar_compounds = get_similar_compounds({names[1]}, 20)
                for compound in similar_compounds[1:]:  # start from index 1 to skip the first similar compound
                    target_similar_compounds_test_query = f"""
                    MATCH (p:Drug)
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
    unique_source_paths = []
    unique_target_paths = []
    unique_graph_rels = set()

    unique_rel_paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result]

    for record in result:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_relationship_string(path_nodes, path_relationships)
        unique_graph_rels.add(rel_string)
    
    source_paths, target_paths, source_relationships, target_relationships = get_source_and_target_paths(graph, names)

    for path in source_paths:
        unique_source_paths.append(path)
    # Construct and add the target path relationship strings to the list
    for path in target_paths:
        unique_target_paths.append(path)

    for rel in source_relationships:
        unique_graph_rels.add(rel)

    for rel in target_relationships:
        unique_graph_rels.add(rel)

    # Convert unique_relationships set to list
    unique_source_paths_list = list(unique_source_paths)
    unique_target_paths_list = list(unique_target_paths)
    unique_graph_rels_list = list(unique_graph_rels)

    return unique_rel_paths, unique_target_paths_list, unique_source_paths_list, unique_graph_rels_list

def query_inter_relationships_direct1(graph: Graph, node:str) -> Tuple[List[Dict[str, Any]], List[str], List[str], Set[str], List[str]]:
    node_label, node_name = get_node_label(graph, node)
    all_nodes = set()
    graph_strings = set()
    relationships_direct = set()
    og_relationships_direct_list = []
    direct_nodes = []

    direct_relationships_query = f"""
    MATCH path=(n:{node_label})-[r]-(m)
    WHERE n.name = "{node_name}" AND n.name IS NOT NULL
    WITH nodes(path) AS nodes, relationships(path) AS rels
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    result_direct = list(graph.run(direct_relationships_query, node=node))
    direct_nodes.extend([node for record in result_direct for node in record['path_nodes']])
    print(direct_relationships_query)

    for record in result_direct:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        rel_string = construct_path_string(path_nodes, path_relationships)
        graph_string = construct_relationship_string(path_nodes, path_relationships)
        graph_strings.add(graph_string)
        relationships_direct.add(rel_string)
        all_nodes.update(record['path_nodes'])
        og_relationships_direct_list.append({'nodes': path_nodes, 'relationships': path_relationships})

    graph_strings_list = list(graph_strings)
    og_relationships_direct_list = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result_direct]
    return og_relationships_direct_list, graph_strings_list, all_nodes, direct_nodes

def query_inter_relationships_between_direct(graph: Graph, direct_nodes, nodes:List[str]) -> str:
    all_node_names = list(nodes) + direct_nodes
    node_labels = get_node_labels(graph, all_node_names)
    unique_labels = list(set(node_labels.values()))
    
    query_parameters_2 = {"nodes": list(nodes) + direct_nodes, "unique_labels": unique_labels}
    total_nodes = list(nodes) + direct_nodes
    print("number of direct nodes")
    print(len(total_nodes))
    # Query for paths between the nodes from the original list

    inter_between_direct_query = """
    MATCH (n)
    WHERE n.name IN $nodes AND any(label in labels(n) WHERE label IN $unique_labels)
    CALL apoc.path.spanningTree(n, {minLevel: 1, limit: 200}) YIELD path
    WITH nodes(path) AS nodes, relationships(path) AS rels
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    result_inter_direct = list(graph.run(inter_between_direct_query, **query_parameters_2))
    print(inter_between_direct_query)

    all_nodes = set()
    graph_strings = set()

    for record in result_inter_direct:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        graph_string = construct_relationship_string(path_nodes, path_relationships)
        graph_strings.add(graph_string)
        all_nodes.update(record['path_nodes'])
    relationships_inter_direct_list = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result_inter_direct]
    graph_strings_list = list(graph_strings)
    print("number of inter direct relations:")
    print(len(relationships_inter_direct_list))
    return relationships_inter_direct_list, graph_strings_list, all_nodes
