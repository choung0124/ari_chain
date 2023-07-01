from streamlit_agraph import agraph, Node, Edge

def draw_graph(paths):
    # Parse relationships
    nodes, edges = parse_relationships(paths)

    # Create node and edge objects
    node_objects = [Node(id=node, label=node) for node in nodes]
    edge_objects = [Edge(source=edge[0], target=edge[1], label=edge[2]) for edge in edges]

    # Set graph configuration
    config = {
        "height": "500px",
        "width": "100%",
        "nodeHighlightBehavior": True,
        "highlightColor": "blue",
        "directed": True,
        "node": {"color": "lightblue", "labelProperty": "label"},
        "link": {"labelProperty": "label", "renderLabel": True},
    }

    # Draw the graph
    agraph(nodes=node_objects, edges=edge_objects, config=config)

def parse_relationships(relationships):
    nodes = []
    edges = []
    for relationship in relationships:
        elements = relationship.split(' -> ')
        for i in range(0, len(elements) - 1, 2):
            source = elements[i]
            relationship_type = elements[i + 1]
            target = elements[i + 2]
            nodes.append(source)
            nodes.append(target)
            edges.append((source, target, relationship_type))
    return list(set(nodes)), edges