import requests, pandas as pd, time
from datetime import datetime,timezone

def get_binance_1m(symbol="BTCUSDT", start="2025-01-01", end="2025-12-31"):
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.strptime(start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime.strptime(end, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)+ (24*60*60*1000 - 1)
    all_data = []

    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000
        }
        r = requests.get(url, params=params)
        data = r.json()
        if not data: break
        all_data.extend(data)
        start_ms = data[-1][0] + 60000  # avanza 1 minuto
        time.sleep(0.1)  # evita rate limit

    df = pd.DataFrame(all_data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms",utc=True)
    df[["open","high","low","close","volume"]] = \
        df[["open","high","low","close","volume"]].astype(float)
    return df[["open_time","open","high","low","close","volume","trades"]]

df = get_binance_1m("BTCUSDT", "2025-01-01", "2025-12-31")
df.to_csv("btc_1m_2025.csv", index=False)
print(df.shape)
print(df["open_time"].min())
print(df["open_time"].max())