"""
This module creates interactive Plotly visualizations of asset correlation graphs:
- Visualizes nodes using return/volatility positions
- Colors edges by positive/negative correlation
- Adds tooltips with return and volatility
"""


import plotly.graph_objects as go

def visualize_financial_graph(G, pos, log_returns, date, filename="graph.html"):
    """
    Visualizes a financial graph using Plotly:
    - Nodes are placed using (avg_return, volatility)
    - Edges are colored green (positive corr) or red (negative corr)
    - Node tooltips show return and volatility

    Args:
        G (networkx.Graph): The correlation graph.
        pos (dict): Node positions from financial features.
        log_returns (pd.DataFrame): Full log return data.
        date (str): The current graph date (YYYY-MM-DD).
        filename (str): Output HTML file name.
    """

    # Extract edge positions and colors
    edge_x, edge_y, edge_colors = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = d['weight']
        color = 'green' if weight > 0 else 'red'
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_colors.append(color)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        mode='lines',
        hoverinfo='none'
    )

    # Node info
    node_x, node_y, node_text = [], [], []
    window_data = log_returns.loc[:date].tail(30)
    avg_returns = window_data.mean()
    volatility = window_data.std()

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        tooltip = f"{node}<br>Return: {avg_returns[node]:.2%}<br>Volatility: {volatility[node]:.2%}"
        node_text.append(tooltip)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            size=20,
            color='skyblue',
            line=dict(width=2)
        )
    )

    # Build the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Correlation Network on {date}",
            titlefont_size=20,
            xaxis_title="Avg Return",
            yaxis_title="Volatility",
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False)
        )
    )

    fig.write_html(filename)


# tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
# prices = download_price_data(tickers)
# log_returns = compute_log_returns(prices)
# rolling_corrs = compute_rolling_correlation(log_returns, window=30)
# corr_matrix = rolling_corrs.loc["2019-01-02"]
# G = build_graph(corr_matrix, threshold=0.6)
#pos = compute_financial_positions(log_returns, date="2019-01-02", window=30)
#visualize_financial_graph(G, pos, log_returns, date="2019-01-02", filename="graph.html")