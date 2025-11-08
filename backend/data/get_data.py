import yfinance as yf

stocks = ["TCS.NS", "TATAMOTORS.NS"]
for ticker in stocks:
    data = yf.download(ticker, period="2y", interval="1d")
    data.to_csv(f"{ticker}.csv")
    print(f"Saved {ticker}.csv")
