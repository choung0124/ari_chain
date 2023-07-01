import networkx as nx
import plotly.graph_objects as go
def draw_graph(paths):
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for path in paths:
        elements = path.split(' -> ')  # Split each path into nodes and relationships
        nodes = elements[::2]  # Select every second element starting from 0 (the nodes)
        relationships = elements[1::2]  # Select every second element starting from 1 (the relationships)
        G.add_nodes_from(nodes)
        for i in range(len(nodes)-1):
            G.add_edge(nodes[i], nodes[i+1], relationship=relationships[i])
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    pos = nx.spring_layout(G)
    node_labels = [node for node in G.nodes()]
    edge_trace = go.Scatter(
        x=[pos[edge[0]][0] for edge in G.edges()] + [None] + [pos[edge[1]][0] for edge in G.edges()],
        y=[pos[edge[0]][1] for edge in G.edges()] + [None] + [pos[edge[1]][1] for edge in G.edges()],
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=[G.edges[edge]['relationship'] for edge in G.edges()],
        textposition='middle center'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=node_labels,  # Add node labels here
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))


    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Network graph made with Python',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig


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