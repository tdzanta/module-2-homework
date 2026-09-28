Module 2 homework (2026)
========================

Run from this folder in PowerShell:

```powershell
.\.python\python.exe hw3.py
```

Python and the required packages are already installed locally in `.python`.
For another Python installation, run `python -m pip install -r requirements.txt`.

The solutions are in `hw3.py`. The answers and supporting tables are saved in
`results/answers_all.md` and the other files in `results/`. Downloads are cached
in `data/`, including the combined `stocks_df.parquet` dataset.

The calculations use September 11, 2026 as the last price date and adjusted
Close prices, matching the lecture's default Ticker.history behavior. Q2 uses
the homework's exact price-volatility formula, even though the 2026 lecture
uses return volatility. Zero volatility can produce an infinite homework
Sharpe value. Q3 shifts separately within each ticker and enters on its first
available post-offer trading date. A growth ratio of 0.92 means an 8% loss.

IPOScoop is a live source. The filtered universe and available Yahoo histories
can differ from the homework's original counts. Q1 filters the visible File
Date, but the source cannot establish historical withdrawal status exactly.
Unavailable tickers are listed in `results/missing_tickers.csv`.

Q4 treats each oversold row as a separate investment, even when trades overlap.
It does not impose a portfolio budget or deduct trading costs.

The holding-period calculations were checked with synthetic data to verify
that shifts never cross ticker boundaries and missing future returns stay missing.
The optional Q5 discussion is included with the answers.
