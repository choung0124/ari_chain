import networkx as nx
import plotly.graph_objects as go
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

def draw_network_graph(nodes, edges):
    G = nx.DiGraph()

    for node in nodes:
        G.add_node(node)

    for edge in edges:
        G.add_edge(edge[0], edge[1], label=edge[2])

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge[2]['label'])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines')

    edge_trace.text = edge_text

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
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

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{nodes[node]} - # of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Network graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    # Add node labels
    for node in G.nodes():
        fig.add_annotation(
            x=pos[node][0], 
            y=pos[node][1],
            text=node,
            showarrow=False,
            font=dict(
                size=10,
                color="Black"
            ),
            bgcolor="White",
            opacity=0.8
        )

    st.plotly_chart(fig)


def parse_relationships(relationships):
    nodes = []
    edges = []
    for relationship in relationships:
        elements = relationship.split(' -> ')
        for i in range(0, len(elements) - 2, 2):
            source = elements[i]
            relationship_type = elements[i + 1]
            target = elements[i + 2]
            nodes.append(source)
            nodes.append(target)
            edges.append((source, target, relationship_type))
    return list(set(nodes)), edges
