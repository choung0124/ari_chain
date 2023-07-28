from __future__ import annotations
from typing import Any, Dict, List, Optional
from py2neo import Graph
from typing import Tuple, Set

def remove_duplicates(paths):
    seen = {}
    result = []
    for path in paths:
        # Convert path to a tuple so it can be added to a dict
        path_tuple = tuple(path)
        if path_tuple not in seen:
            result.append(path)
            seen[path_tuple] = True
    return result

def get_node_label(graph: Graph, node_name: str) -> str:
    query = """
    CALL apoc.cypher.runTimeboxed(
        "MATCH (node)
        WHERE toLower(node.name) = toLower($nodeName)
        RETURN head(labels(node)) AS FirstLabel, node.name AS NodeName",
        {nodeName: $nodeName},
        60000
    ) YIELD value
    RETURN value.FirstLabel, value.NodeName
    """
    result = graph.run(query, nodeName=node_name).data()
    if result:
        print(query)
        return result[0]['value.FirstLabel'], result[0]['value.NodeName']
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

def get_source_and_target_paths(graph: Graph, names: List[str]) -> Tuple[List[Relationship], List[Relationship]]:
    source_label, source_name = get_node_label(graph, names[0])
    target_label, target_name = get_node_label(graph, names[1])

    query_source = f"""
    MATCH path=(source:{source_label})-[rel*1..2]->(node)
    WHERE source.name = "{source_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    UNION
    MATCH path=(node)-[rel*1..2]->(source:{source_label})
    WHERE source.name = "{source_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    LIMIT 50000
    """

    query_target = f"""
    MATCH path=(source:{target_label})-[rel*1..2]->(node)
    WHERE source.name = "{target_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    UNION
    MATCH path=(node)-[rel*1..2]->(source:{target_label})
    WHERE source.name = "{target_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    LIMIT 50000
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

    return source_paths, target_paths

def find_shortest_paths(graph: Graph, names: List[str]) -> List[Dict[str, Any]]:
    source_label, source_name = get_node_label(graph, names[0])
    target_label, target_name = get_node_label(graph, names[1])

    if source_label is None or target_label is None:
        return None
    
    names_conditions = f'WHERE source.name = "{source_name}" AND target.name = "{target_name}"'
    query = f"""
    MATCH (source:{source_label}), (target:{target_label})
    {names_conditions}
    MATCH p = allShortestPaths((source)-[*]-(target))
    WITH p, [rel IN relationships(p) | type(rel)] AS path_relationships
    WITH relationships(p) AS rels, nodes(p) AS nodes, path_relationships
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    LIMIT 50000
    """
    print(query)
    result = graph.run(query)

    # Initialize a set to store unique associated genes
    final_source_paths = []
    final_target_paths = []

    unique_rel_paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result]
    source_paths, target_paths = get_source_and_target_paths(graph, names)

    for path in source_paths:
        final_source_paths.append(path)
    # Construct and add the target path relationship strings to the list
    for path in target_paths:
        final_target_paths.append(path)
    # Convert unique_relationships set to list

    return unique_rel_paths, final_target_paths, final_source_paths

def query_direct(graph: Graph, node:str) -> Tuple[List[Dict[str, Any]], List[str], List[str], Set[str], List[str]]:
    node_label, node_name = get_node_label(graph, node)
    paths_list = []

    query = f"""
    MATCH path=(source:{node_label})-[rel*1..2]->(node)
    WHERE source.name = "{node_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    UNION
    MATCH path=(node)-[rel*1..2]->(source:{node_label})
    WHERE source.name = "{node_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    LIMIT 50000
    """
    result = list(graph.run(query, node=node))
    print(query)

    for record in result:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        paths_list.append({'nodes': path_nodes, 'relationships': path_relationships})

    return paths_list

def query_between_direct(graph: Graph, direct_nodes, nodes:List[str]) -> str:
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

    paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result_inter_direct]
    print("number of inter direct relations:")
    print(len(paths))

    return paths
