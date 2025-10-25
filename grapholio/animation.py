"""
This module automates the generation of a sequence of correlation graph visualizations
over multiple dates. Each graph is saved as a separate interactive HTML file.
"""

from grapholio.graph_builder import build_graph
from grapholio.features import compute_financial_positions
from grapholio.visualizer import visualize_financial_graph

def animate_graphs_over_time(log_returns, rolling_corrs, dates, threshold=0.6, output_dir="graphs"):
    """
    Loops over a list of dates and generates interactive HTML graph visualizations.

    Args:
        log_returns (pd.DataFrame): Log return data.
        rolling_corrs (pd.DataFrame): MultiIndex correlation matrices.
        dates (list of str): List of dates (YYYY-MM-DD) to generate graphs for.
        threshold (float): Correlation threshold for graph edges.
        output_dir (str): Directory to save HTML files in (default: 'graphs').
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for date in dates:
        print(f"Rendering: {date}")
        try:
            corr_matrix = rolling_corrs.loc[date]
            G = build_graph(corr_matrix, threshold=threshold)
            pos = compute_financial_positions(log_returns, date)
            filename = os.path.join(output_dir, f"correlation_graph_{date}.html")
            visualize_financial_graph(G, pos, log_returns, date, filename)
        except KeyError:
            print(f"Skipped: {date} (missing correlation)")



# import pandas as pd
# tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
# prices = download_price_data(tickers)
# log_returns = compute_log_returns(prices)
# rolling_corrs = compute_rolling_correlation(log_returns, window=30)
# dates = pd.date_range("2018-01-01", "2018-06-01", freq='MS').strftime('%Y-%m-%d').tolist()
# animate_graphs_over_time(log_returns, rolling_corrs, dates)
