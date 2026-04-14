# Stock Analysis MCP Server

Automated stock analysis tool built on Anthropic's Model Context Protocol (MCP).
Connects Claude Code to live market data via Yahoo Finance.

## What it does
Type one sentence to Claude and get a full quantitative analysis in ~60 seconds:
- Live OHLCV price data across any timeframe
- ARIMA(1,1,1) price forecast with 95% confidence intervals
- RSI, MACD, Bollinger Band technical indicators
- Fundamental analysis: PE, margins, earnings history, analyst targets
- Bear / base / bull investment scenarios for any dollar amount
- Peer comparison across sector competitors

## Demo
> "Run a full analysis on NVDA with $500 invested for 12 weeks"

Claude automatically calls all 5 tools, pulls live data, and returns
a complete investment analysis with scenario modeling.

## Setup
1. Install Python 3.11+
2. `pip install -r requirements.txt`
3. `claude mcp add stock-analysis -- python /path/to/server.py`
4. Open Claude Code and ask about any stock

## Built with
- [FastMCP](https://gofastmcp.com) — MCP server framework
- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance data
- [Anthropic Claude Code](https://claude.ai/code) — AI interface
