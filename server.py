"""
Stock Data MCP Server
Automatically fetches OHLCV, indicators, fundamentals and feeds them to Claude
for stock analysis. Built for use with Claude Code.

Usage:
    python server.py

Requirements:
    pip install mcp yfinance pandas numpy requests fastmcp
"""

import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("stock-analysis-server")


# ── helper functions ────────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)


def compute_macd(prices: pd.Series):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26).round(3)
    signal = macd_line.ewm(span=9, adjust=False).mean().round(3)
    hist = (macd_line - signal).round(3)
    return macd_line, signal, hist


def compute_bollinger(prices: pd.Series, window: int = 20, num_std: float = 2):
    sma = prices.rolling(window).mean().round(2)
    std = prices.rolling(window).std().round(4)
    upper = (sma + num_std * std).round(2)
    lower = (sma - num_std * std).round(2)
    pct_b = ((prices - lower) / (upper - lower) * 100).round(1)
    return sma, upper, lower, pct_b


def compute_arima_forecast(prices: pd.Series, horizon: int = 60):
    """Simple AR(1) on first differences — matches the model used in analysis."""
    diff = prices.diff().dropna()
    mu = float(diff.mean())
    # AR(1) via autocorrelation
    n = len(diff)
    cov = sum((diff.iloc[i] - mu) * (diff.iloc[i-1] - mu) for i in range(1, n))
    var = sum((d - mu) ** 2 for d in diff)
    phi = cov / var if var != 0 else 0
    # Residuals
    resids = []
    for i in range(1, n):
        yhat = mu + phi * (diff.iloc[i-1] - mu)
        resids.append(diff.iloc[i] - yhat)
    sigma = float(np.std(resids))
    last_p = float(prices.iloc[-1])
    last_d = float(diff.iloc[-1])
    last_e = resids[-1]
    forecasts, ci95l, ci95u = [], [], []
    for h in range(1, horizon + 1):
        fd = mu + phi * (last_d - mu) + (last_e if h == 1 else 0)
        fp = round(last_p + fd, 2)
        sd = sigma * (h ** 0.5)
        forecasts.append(fp)
        ci95l.append(round(fp - 1.96 * sd, 2))
        ci95u.append(round(fp + 1.96 * sd, 2))
        last_d = fd
        last_p = fp
        last_e = 0
    return forecasts, ci95l, ci95u, round(phi, 4), round(sigma, 4)


# ── MCP tools ───────────────────────────────────────────────────────────────

@mcp.tool()
def get_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> str:
    """
    Fetch OHLCV price history for a stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'CWEN', 'NVDA')
        period: Time period — '1wk', '1mo', '3mo', '6mo', '1y', '2y', '5y'
        interval: Bar size — '15m', '1h', '1d', '1wk'

    Returns:
        JSON string with dates, OHLCV arrays and basic stats.
    """
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return json.dumps({"error": f"No data found for {ticker}"})
    closes = hist["Close"].round(2)
    result = {
        "ticker": ticker.upper(),
        "period": period,
        "interval": interval,
        "count": len(hist),
        "start": str(hist.index[0].date()),
        "end": str(hist.index[-1].date()),
        "dates": [str(d.date()) for d in hist.index],
        "open":   hist["Open"].round(2).tolist(),
        "high":   hist["High"].round(2).tolist(),
        "low":    hist["Low"].round(2).tolist(),
        "close":  closes.tolist(),
        "volume": hist["Volume"].astype(int).tolist(),
        "stats": {
            "last_close":  round(float(closes.iloc[-1]), 2),
            "52wk_high":   round(float(closes.max()), 2),
            "52wk_low":    round(float(closes.min()), 2),
            "total_return_pct": round((closes.iloc[-1] / closes.iloc[0] - 1) * 100, 2),
            "ann_volatility_pct": round(float(closes.pct_change().std() * np.sqrt(252) * 100), 2),
        }
    }
    return json.dumps(result)


@mcp.tool()
def get_technical_indicators(ticker: str, period: str = "1y") -> str:
    """
    Compute RSI, MACD, Bollinger Bands and ARIMA forecast for a stock.

    Args:
        ticker: Stock ticker symbol
        period: Lookback period for daily data ('1y' recommended, '2y' for more)

    Returns:
        JSON with all indicator arrays plus 60-day ARIMA forecast with 95% CI.
    """
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period=period, interval="1d")
    if hist.empty:
        return json.dumps({"error": f"No data for {ticker}"})
    closes = hist["Close"].round(2)
    dates = [str(d.date()) for d in hist.index]

    rsi = compute_rsi(closes)
    macd_line, signal_line, macd_hist = compute_macd(closes)
    sma20, bb_up, bb_lo, pct_b = compute_bollinger(closes)
    fc, ci95l, ci95u, phi, sigma = compute_arima_forecast(closes, horizon=60)

    # Sharpe (simple, 0% risk-free)
    rets = closes.pct_change().dropna()
    sharpe = round(float(rets.mean() / rets.std() * np.sqrt(252)), 3)

    result = {
        "ticker": ticker.upper(),
        "dates": dates,
        "rsi14": [None if pd.isna(v) else v for v in rsi.tolist()],
        "macd_line":   [None if pd.isna(v) else v for v in macd_line.tolist()],
        "macd_signal": [None if pd.isna(v) else v for v in signal_line.tolist()],
        "macd_hist":   [None if pd.isna(v) else v for v in macd_hist.tolist()],
        "bb_sma20":  [None if pd.isna(v) else v for v in sma20.tolist()],
        "bb_upper":  [None if pd.isna(v) else v for v in bb_up.tolist()],
        "bb_lower":  [None if pd.isna(v) else v for v in bb_lo.tolist()],
        "bb_pct_b":  [None if pd.isna(v) else v for v in pct_b.tolist()],
        "arima_forecast": {
            "phi": phi,
            "sigma_daily": sigma,
            "ann_vol": round(sigma * np.sqrt(252), 2),
            "sharpe_1yr": sharpe,
            "horizon_days": 60,
            "point_forecast": fc,
            "ci95_lower": ci95l,
            "ci95_upper": ci95u,
            "day12_forecast": fc[11],
            "day12_ci95": [ci95l[11], ci95u[11]],
            "day60_forecast": fc[59],
            "day60_ci95": [ci95l[59], ci95u[59]],
        },
        "current_signals": {
            "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
            "macd_hist_latest": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else None,
            "bb_pct_b_latest": float(pct_b.iloc[-1]) if not pd.isna(pct_b.iloc[-1]) else None,
            "price_vs_sma20": round((float(closes.iloc[-1]) / float(sma20.iloc[-1]) - 1) * 100, 2),
        }
    }
    return json.dumps(result)


@mcp.tool()
def get_fundamentals(ticker: str) -> str:
    """
    Fetch fundamental data: valuation, earnings estimates, analyst targets,
    dividend info, institutional ownership, and peer comparison.

    Args:
        ticker: Stock ticker symbol

    Returns:
        JSON with all fundamental metrics.
    """
    stock = yf.Ticker(ticker.upper())
    info = stock.info
    if not info:
        return json.dumps({"error": f"No fundamental data for {ticker}"})

    def safe(key, default=None):
        return info.get(key, default)

    # Analyst targets
    try:
        targets = stock.analyst_price_targets
        target_data = {
            "mean": round(float(targets.get("mean", 0)), 2),
            "high": round(float(targets.get("high", 0)), 2),
            "low":  round(float(targets.get("low", 0)), 2),
        } if targets else {}
    except Exception:
        target_data = {}

    # Earnings history
    try:
        earnings = stock.earnings_history
        earnings_list = []
        if earnings is not None and not earnings.empty:
            for _, row in earnings.tail(8).iterrows():
                earnings_list.append({
                    "date": str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
                    "actual": round(float(row.get("epsActual", 0)), 3),
                    "estimate": round(float(row.get("epsEstimate", 0)), 3),
                    "surprise_pct": round(float(row.get("surprisePercent", 0)) * 100, 1),
                })
    except Exception:
        earnings_list = []

    result = {
        "ticker": ticker.upper(),
        "company_name": safe("longName"),
        "sector": safe("sector"),
        "industry": safe("industry"),
        "market_cap": safe("marketCap"),
        "valuation": {
            "trailing_pe":   round(safe("trailingPE", 0), 2),
            "forward_pe":    round(safe("forwardPE", 0), 2),
            "price_to_sales": round(safe("priceToSalesTrailing12Months", 0), 2),
            "price_to_book":  round(safe("priceToBook", 0), 2),
            "ev_to_ebitda":   round(safe("enterpriseToEbitda", 0), 2),
        },
        "growth": {
            "revenue_growth":  safe("revenueGrowth"),
            "earnings_growth": safe("earningsGrowth"),
            "ltg_forecast":    safe("longTermPotentialGrowthRate"),
        },
        "profitability": {
            "gross_margin":   safe("grossMargins"),
            "net_margin":     safe("profitMargins"),
            "roe":            safe("returnOnEquity"),
            "roa":            safe("returnOnAssets"),
        },
        "financial_health": {
            "current_ratio":      safe("currentRatio"),
            "debt_to_equity":     safe("debtToEquity"),
            "interest_coverage":  safe("interestCoverage"),
            "free_cashflow":      safe("freeCashflow"),
        },
        "dividend": {
            "yield_pct":        round((safe("dividendYield") or 0) * 100, 2),
            "annual_dividend":  safe("dividendRate"),
            "payout_ratio":     safe("payoutRatio"),
            "ex_dividend_date": str(safe("exDividendDate", "")),
            "dividend_growth":  safe("fiveYearAvgDividendYield"),
        },
        "analyst": {
            "recommendation": safe("recommendationKey"),
            "num_analysts":   safe("numberOfAnalystOpinions"),
            "price_targets":  target_data,
            "current_price":  round(safe("currentPrice", 0), 2),
        },
        "ownership": {
            "institutional_pct": round((safe("heldPercentInstitutions") or 0) * 100, 1),
            "insider_pct":       round((safe("heldPercentInsiders") or 0) * 100, 1),
            "short_interest_pct": round((safe("shortPercentOfFloat") or 0) * 100, 2),
        },
        "earnings_history": earnings_list,
    }
    return json.dumps(result)


@mcp.tool()
def get_investment_scenario(
    ticker: str,
    investment_usd: float,
    horizon_weeks: int = 12
) -> str:
    """
    Run a full investment scenario analysis for a given dollar amount.
    Combines price data, ARIMA forecast, fundamentals, and analyst targets
    to produce bear/base/bull scenarios with P&L for each.

    Args:
        ticker: Stock ticker symbol
        investment_usd: Dollar amount to invest (e.g. 150.0)
        horizon_weeks: Investment horizon in weeks (default 12)

    Returns:
        JSON with full scenario analysis including shares, P&L, and expected value.
    """
    # Get price and fundamentals
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period="1y", interval="1d")
    if hist.empty:
        return json.dumps({"error": f"No data for {ticker}"})

    closes = hist["Close"].round(2)
    last_price = float(closes.iloc[-1])
    shares = round(investment_usd / last_price, 4)

    # ARIMA forecast
    horizon_days = horizon_weeks * 5  # trading days
    fc, ci95l, ci95u, phi, sigma = compute_arima_forecast(closes, horizon=max(horizon_days, 60))
    fc_price = fc[horizon_days - 1]
    ci_lower = ci95l[horizon_days - 1]
    ci_upper = ci95u[horizon_days - 1]

    # Dividend
    info = stock.info
    annual_div = info.get("dividendRate") or 0
    weeks_per_year = 52
    div_in_period = round(annual_div * (horizon_weeks / weeks_per_year) * shares, 2)

    # Analyst targets
    try:
        tgt = stock.analyst_price_targets
        analyst_mean = float(tgt.get("mean", fc_price))
        analyst_high = float(tgt.get("high", ci_upper))
        analyst_low  = float(tgt.get("low", ci_lower))
    except Exception:
        analyst_mean = fc_price
        analyst_high = ci_upper
        analyst_low  = ci_lower

    def scenario(price, label, basis):
        val = round(shares * price, 2)
        pl  = round(val - investment_usd, 2)
        total = round(pl + div_in_period, 2)
        ret_pct = round(total / investment_usd * 100, 2)
        return {
            "scenario": label,
            "basis": basis,
            "price_per_share": round(price, 2),
            "position_value": val,
            "capital_pl": pl,
            "dividend_income": div_in_period,
            "total_return_usd": total,
            "total_return_pct": ret_pct,
        }

    scenarios = [
        scenario(ci_lower,      "Deep bear",   "ARIMA 95% CI lower"),
        scenario(analyst_low,   "Bear",        "Analyst low target"),
        scenario(fc_price,      "Base case",   "ARIMA point forecast"),
        scenario(analyst_mean,  "Bull",        "Analyst mean target"),
        scenario(analyst_high,  "Strong bull", "Analyst high target"),
        scenario(ci_upper,      "Deep bull",   "ARIMA 95% CI upper"),
    ]

    # Expected value (simple weighted average skewed to base)
    weights = [0.05, 0.15, 0.40, 0.25, 0.10, 0.05]
    ev_price = sum(w * s["price_per_share"] for w, s in zip(weights, scenarios))
    ev_value = round(shares * ev_price + div_in_period, 2)
    ev_return = round((ev_value - investment_usd) / investment_usd * 100, 2)

    result = {
        "ticker":          ticker.upper(),
        "investment_usd":  investment_usd,
        "current_price":   last_price,
        "shares":          shares,
        "horizon_weeks":   horizon_weeks,
        "arima_params":    {"phi": phi, "sigma_daily": sigma},
        "scenarios":       scenarios,
        "expected_value":  {
            "price":       round(ev_price, 2),
            "position_value": ev_value,
            "total_return_usd": round(ev_value - investment_usd, 2),
            "total_return_pct": ev_return,
        },
        "dividend_estimate": div_in_period,
    }
    return json.dumps(result)


@mcp.tool()
def get_peer_comparison(ticker: str) -> str:
    """
    Fetch peer/competitor tickers and compare key metrics side by side.

    Args:
        ticker: The primary stock ticker to compare

    Returns:
        JSON with side-by-side valuation, growth and risk metrics for the stock and peers.
    """
    stock = yf.Ticker(ticker.upper())
    info = stock.info
    peers_raw = info.get("companyOfficers", [])  # not reliable for peers

    # Use sector-based common peers for well-known stocks
    peer_map = {
        "CWEN":  ["NEP", "AES", "BEP", "XIFR", "NRG"],
        "AAPL":  ["MSFT", "GOOGL", "META", "AMZN"],
        "NVDA":  ["AMD", "INTC", "QCOM", "TSM"],
        "TSLA":  ["GM", "F", "RIVN", "NIO"],
        "SPY":   ["QQQ", "IWM", "DIA"],
        "AMZN":  ["MSFT", "GOOGL", "META", "AAPL"],
    }

    peer_tickers = peer_map.get(ticker.upper(), [])
    all_tickers = [ticker.upper()] + peer_tickers

    rows = []
    for t in all_tickers:
        try:
            i = yf.Ticker(t).info
            rows.append({
                "ticker":       t,
                "name":         i.get("shortName", t),
                "market_cap":   i.get("marketCap"),
                "trailing_pe":  round(i.get("trailingPE") or 0, 1),
                "forward_pe":   round(i.get("forwardPE") or 0, 1),
                "price_to_sales": round(i.get("priceToSalesTrailing12Months") or 0, 2),
                "div_yield_pct": round((i.get("dividendYield") or 0) * 100, 2),
                "net_margin":   round((i.get("profitMargins") or 0) * 100, 2),
                "revenue_growth": round((i.get("revenueGrowth") or 0) * 100, 2),
                "recommendation": i.get("recommendationKey", "n/a"),
            })
        except Exception:
            continue

    return json.dumps({"primary": ticker.upper(), "peers": rows})


if __name__ == "__main__":
    print("Starting Stock Analysis MCP Server...")
    print("Tools available:")
    print("  - get_price_history")
    print("  - get_technical_indicators")
    print("  - get_fundamentals")
    print("  - get_investment_scenario")
    print("  - get_peer_comparison")
    mcp.run(transport="stdio")
