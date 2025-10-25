"""
This is the main script to run your entire financial graph visualization pipeline:
- Downloads price data
- Computes log returns and rolling correlations
- Builds graph layouts using financial features (return & volatility)
- Visualizes correlation graphs over time (animated HTML series)
"""

import pandas as pd
from grapholio.data_loader import download_price_data, compute_log_returns
from grapholio.graph_builder import compute_rolling_correlation
from grapholio.animation import animate_graphs_over_time

def main():
    # Configuration
    tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
    start_date = "2015-01-01"
    end_date = "2020-01-01"
    corr_window = 30
    corr_threshold = 0.6
    output_dir = "graphs"

    print("📥 Downloading price data...")
    prices = download_price_data(tickers, start=start_date, end=end_date)

    print("📈 Calculating log returns...")
    log_returns = compute_log_returns(prices)

    print("🔁 Computing rolling correlations...")
    rolling_corrs = compute_rolling_correlation(log_returns, window=corr_window)

    print("🗓️ Preparing timeline...")
    dates = pd.date_range("2018-01-01", "2018-06-01", freq='MS').strftime('%Y-%m-%d').tolist()

    print("🎞️ Generating animated graphs...")
    animate_graphs_over_time(log_returns, rolling_corrs, dates, threshold=corr_threshold, output_dir=output_dir)

    print(f"✅ Done! HTML files saved in: {output_dir}/")

if __name__ == "__main__":
    main()
