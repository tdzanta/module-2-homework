"""Module 2 homework (2026). Saved data goes in data/, answers in results/."""

from io import StringIO
from pathlib import Path
import re

import gdown
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
AS_OF = pd.Timestamp("2026-09-11")
GROWTH = [f"future_growth_{m}_m" for m in range(1, 13)]


def read_table(url, filename):
    path = DATA / filename
    if not path.exists():
        response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    table = pd.read_html(StringIO(path.read_text(encoding="utf-8")))[0]
    table.columns = [" ".join(str(c).split()) for c in table.columns]
    return table


def number(value):
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[$,%\s]", "", str(value))
    return pd.to_numeric(cleaned, errors="coerce")


def company_type(name):
    rules = [
        ("Technologies", ["Technologies"]),
        ("Acquisition Corp", ["Acquisition Corp", "Acquisition Corporation", "Corp"]),
        ("Inc.", ["Inc", "Incorporated"]),
        ("Group", ["Group"]),
        ("Limited", ["Ltd", "Limited"]),
        ("Holdings", ["Holdings", "Holding"]),
    ]
    for label, patterns in rules:
        if any(pattern in str(name) for pattern in patterns):
            return label
    return "Other"


def question_1():
    filed = read_table("https://www.iposcoop.com/ipos-recently-filed/", "recently_filed.html")
    dates = pd.to_datetime(filed["File Date"], errors="coerce")
    withdrawn = filed.loc[
        filed["Expected To Trade"].str.strip().eq("Withdrawn") & dates.lt(AS_OF)
    ].copy()
    withdrawn["Company Type"] = withdrawn["Company"].map(company_type)
    for col in ["Price Low", "Price High", "Shares (millions)", "Est $ Vol (millions)"]:
        withdrawn[col] = withdrawn[col].map(number)
    withdrawn["Avg_price"] = (withdrawn["Price Low"] + withdrawn["Price High"]) / 2
    withdrawn["Shares_offered_value"] = (
        withdrawn["Shares (millions)"] * withdrawn["Avg_price"]
    ).fillna(withdrawn["Est $ Vol (millions)"])
    totals = withdrawn.groupby("Company Type")["Shares_offered_value"].sum(min_count=1).sort_values(ascending=False)
    withdrawn.to_csv(OUT / "q1_withdrawn.csv", index=False)
    totals.to_csv(OUT / "q1_totals.csv")
    return (f"Q1: {totals.index[0]}, ${totals.iloc[0]:,.3f} million; {len(withdrawn)} rows.\n"
            "The live table is filtered by File Date before September 11. It does not provide "
            "historical withdrawal status, so this is not an exact archived September 11 snapshot.")


def download_stocks():
    ipos = read_table("https://www.iposcoop.com/2025-pricings/", "2025_pricings.html")
    dates = pd.to_datetime(ipos["Offer Date"], errors="coerce")
    return_col = next(c for c in ipos.columns if "return" in c.lower())
    returns = ipos[return_col].map(number)
    selected = ipos.loc[dates.lt("2025-09-01") & returns.notna() & returns.ne(0)].copy()
    selected["Offer Date"] = dates.loc[selected.index]
    selected.to_csv(OUT / "q2_filtered_ipos.csv", index=False)
    ticker_col = "Symbol" if "Symbol" in selected else "Ticker"
    frames, failures = [], []
    yf.set_tz_cache_location(str(DATA / "yfinance_cache"))
    for i, (_, row) in enumerate(selected.drop_duplicates(ticker_col).iterrows(), 1):
        ticker = str(row[ticker_col]).strip()
        yahoo_ticker = ticker.replace(".", "-")
        cache = DATA / f"prices_{yahoo_ticker}.parquet"
        try:
            if cache.exists():
                prices = pd.read_parquet(cache)
            else:
                prices = yf.Ticker(yahoo_ticker).history(
                    start=row["Offer Date"].strftime("%Y-%m-%d"),
                    end="2026-09-12", auto_adjust=True,
                )
                if prices.empty:
                    raise ValueError("No price history returned")
                if isinstance(prices.columns, pd.MultiIndex):
                    prices.columns = prices.columns.get_level_values(0)
                prices.index = pd.to_datetime(prices.index).tz_localize(None)
                prices = prices.rename_axis("Date").reset_index()
                prices.to_parquet(cache, index=False)
            prices = prices.loc[prices["Date"].between(row["Offer Date"], AS_OF)].copy()
            prices = prices.dropna(subset=["Close"]).sort_values("Date")
            prices = prices.loc[prices["Close"].gt(0)].drop_duplicates("Date")
            if prices.empty:
                raise ValueError("No valid post-offer prices")
            prices["Ticker"] = ticker
            frames.append(prices)
        except Exception as exc:
            failures.append({"Ticker": ticker, "reason": str(exc)})
        print(f"{i}/{len(selected)} {ticker}", flush=True)
    pd.DataFrame(failures, columns=["Ticker", "reason"]).to_csv(OUT / "missing_tickers.csv", index=False)
    if not frames:
        raise ValueError("No stocks downloaded; see missing_tickers.csv")
    print(f"Selected {len(selected)} IPOs; downloaded {len(frames)} tickers.", flush=True)
    return pd.concat(frames, ignore_index=True)


def add_features(stocks_df):
    stocks_df = stocks_df.sort_values(["Ticker", "Date"]).copy()
    close = stocks_df.groupby("Ticker")["Close"]
    stocks_df["growth_252d"] = stocks_df["Close"] / close.shift(252)
    # Use the volatility formula given in Q2.
    stocks_df["volatility"] = close.transform(lambda s: s.rolling(30).std()) * np.sqrt(252)
    stocks_df["Sharpe"] = (stocks_df["growth_252d"] - 0.05) / stocks_df["volatility"]
    for month, column in enumerate(GROWTH, 1):
        stocks_df[column] = close.shift(-21 * month) / stocks_df["Close"]
    return stocks_df


def questions_2_and_3(stocks_df):
    stocks_df.to_parquet(DATA / "stocks_df.parquet", index=False)
    snapshot = stocks_df.loc[stocks_df["Date"].eq(AS_OF)]
    stats = snapshot[["growth_252d", "volatility", "Sharpe"]].describe()
    stats.to_csv(OUT / "q2_statistics.csv")
    snapshot.to_csv(OUT / "q2_snapshot.csv", index=False)
    if snapshot["Sharpe"].notna().sum() == 0:
        q2 = "Q2: No valid Sharpe values on 2026-09-11; cannot compute the requested median."
    else:
        q2 = (f"Q2: Median homework Sharpe = {snapshot['Sharpe'].median():.6f}. "
              f"{snapshot['growth_252d'].count()} stocks reached 252 trading days. "
              f"Growth mean = {snapshot['growth_252d'].mean():.6f}; "
              f"median = {snapshot['growth_252d'].median():.6f}. "
              f"Zero-volatility rows: {snapshot['volatility'].eq(0).sum()}; "
              "the exact homework formula can produce infinite Sharpe values for those rows.")
    first_days = stocks_df.groupby("Ticker", as_index=False)["Date"].min().rename(columns={"Date": "min_date"})
    entries = first_days.merge(stocks_df, left_on=["Ticker", "min_date"],
                               right_on=["Ticker", "Date"], how="inner", validate="one_to_one")
    summary = entries[GROWTH].describe().T
    summary["median_return_pct"] = (summary["50%"] - 1) * 100
    summary["mean_return_pct"] = (summary["mean"] - 1) * 100
    summary.to_csv(OUT / "q3_holding_periods.csv")
    entries.to_csv(OUT / "q3_ipo_entries.csv", index=False)
    # Check the same stocks across all 12 holding periods as well.
    entries.dropna(subset=GROWTH)[GROWTH].describe().T.to_csv(OUT / "q3_complete_cohort.csv")
    if summary["50%"].notna().sum() == 0:
        return q2, "Q3: No observed holding-period returns."
    best = summary["50%"].idxmax()
    month = int(best.split("_")[2])
    median = summary.loc[best, "50%"]
    unit = "month" if month == 1 else "months"
    q3 = (f"Q3: Best holding period = {month} {unit}; median growth = {median:.6f} "
          f"({(median - 1) * 100:.2f}% return). Mean growth at that horizon = "
          f"{summary.loc[best, 'mean']:.6f}.\n"
          f"Median growth at 1 and 12 months: {summary.iloc[0]['50%']:.6f}, "
          f"{summary.iloc[-1]['50%']:.6f}.\n"
          "The mean is higher than the median at the best holding period, so a few big "
          "winners make the average look better than the typical stock's result. "
          "The median generally falls as the holding period gets longer, although it "
          "does not fall every month. Even the best median is below 1, so buying every "
          "IPO and holding for this period still loses money on the typical stock. "
          "Stocks without enough future prices are left out of that month's calculation.")
    return q2, q3


def question_4():
    path = DATA / "data.parquet"
    if not path.exists():
        gdown.download(id="1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-", output=str(path), quiet=False)
    schema = pq.read_schema(path).names
    date_col = next(c for c in schema if c.lower() == "date")
    rsi_col = next(c for c in schema if c.lower() == "rsi")
    # These are the only columns needed for Q4.
    df = pd.read_parquet(path, columns=[date_col, rsi_col, "growth_future_30d"])
    dates = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    selected_df = df.loc[dates.between("2000-01-01", "2025-06-01") & df[rsi_col].lt(30)]
    returns = selected_df["growth_future_30d"].dropna() - 1
    if returns.empty:
        raise ValueError("No RSI trades with observed forward returns")
    net_income = 1000 * returns.sum()
    result = {"signals": len(selected_df), "completed_trades": len(returns),
              "missing_returns": len(selected_df) - len(returns),
              "mean_return": returns.mean(), "win_rate": returns.gt(0).mean(),
              "net_income_dollars": net_income, "net_income_thousands": net_income / 1000}
    pd.Series(result).to_csv(OUT / "q4_results.csv")
    return (f"Q4: Net income = ${net_income:,.2f}, or {net_income / 1000:.6f} thousand dollars. "
            f"{len(selected_df):,} signals, {len(returns):,} observed outcomes; "
            f"mean return {returns.mean():.2%}, win rate {returns.gt(0).mean():.2%}. "
            "Each signal is a separate $1,000 investment, including overlapping trades; "
            "this is gross profit before fees, not a compounded portfolio return.")


Q5 = """Q5: I would be more selective about which IPOs to buy. I would test companies
with growing revenue, manageable debt and reasonable valuations, and compare buying
on the first day with waiting for the price to settle. I would also limit how much
goes into any one stock.

To check whether this helps, I would develop the rules on older IPOs and test them
on a later group without changing the rules. I would include trading costs and
delisted stocks, and use only information that was available when buying. The aim
would be to improve the typical return, rather than rely on a few huge winners.
These ideas would need testing before claiming they improve profitability."""


def main():
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    answer_1 = question_1()
    stocks_path = DATA / "stocks_df.parquet"
    if stocks_path.exists():
        stocks_df = pd.read_parquet(stocks_path)
    else:
        stocks_df = download_stocks()

    stocks_df = add_features(stocks_df)
    answer_2, answer_3 = questions_2_and_3(stocks_df)
    answer_4 = question_4()

    answers = "\n\n".join([answer_1, answer_2, answer_3, answer_4, Q5])
    print(answers)
    (OUT / "answers_all.md").write_text(answers + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
