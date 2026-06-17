import yfinance as yf

dxy = yf.download(
    "DX-Y.NYB",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

gold = yf.download(
    "GC=F",
    period="6mo",
    interval="1h",
    auto_adjust=True,
)

print("\nDXY Rows:", len(dxy))
print("Gold Rows:", len(gold))

print("\nLatest DXY Close:")
print(dxy["Close"].tail())

print("\nLatest Gold Close:")
print(gold["Close"].tail())
