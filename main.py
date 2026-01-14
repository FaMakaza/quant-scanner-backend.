import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yfinance as yf
import pandas as pd
import asyncio
import os
from datetime import datetime

class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}
        # A focused list to ensure fast initial scans and no rate limits
        self.symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "GC=F", "CL=F", "BTC-USD", "SPY"]

    def calculate_bias(self, df):
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df['Close']
            ema = close.ewm(span=20, adjust=False).mean()
            curr = float(close.iloc[-1])
            if curr > float(ema.iloc[-1]): return 1
            if curr < float(ema.iloc[-1]): return -1
            return 0
        except: return 0

    def analyze(self, symbol):
        try:
            # High-efficiency download: 1 request for 4 timeframes
            raw = yf.download(symbol, period="1y", interval="1h", progress=False, auto_adjust=True)
            if raw.empty: return None

            # Resample to get MTF data
            tf_data = {
                "W1": raw.resample('W').last(),
                "D1": raw.resample('D').last(),
                "H4": raw.resample('4H').last(),
                "H1": raw
            }

            biases = {tf: self.calculate_bias(tf_data[tf]) for tf in self.weights.keys()}
            score = sum(biases[tf] * self.weights[tf] for tf in self.weights)
            
            # Status and Alignment
            status = "BULLISH" if score >= 0.2 else "BEARISH" if score <= -0.2 else "NEUTRAL"
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for v in biases.values() if v == direction) if direction != 0 else 0
            
            # M15 Entry Logic
            m15_raw = yf.download(symbol, period="1d", interval="15m", progress=False, auto_adjust=True)
            price = float(m15_raw['Close'].iloc[-1]) if not m15_raw.empty else 0
            
            return {
                "symbol": symbol.replace("=X", "").replace("=F", ""),
                "price": round(price, 4),
                "status": status,
                "biases": biases,
                "risk_tier": "A+" if alignment == 4 else "A" if alignment == 3 else "B",
                "signal": status if alignment >= 3 else "WAITING", # Blue dot if waiting, Green/Red if A/A+
                "alignment_val": alignment
            }
        except: return None

engine = QuantEngine()
MARKET_STATE = {}
PROGRESS = {"current": 0, "total": len(engine.symbols)}

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

async def scanner_loop():
    while True:
        PROGRESS["current"] = 0
        for symbol in engine.symbols:
            res = engine.analyze(symbol)
            if res:
                MARKET_STATE[symbol] = res
            PROGRESS["current"] += 1
            await asyncio.sleep(1) # Gentle pacing
        await asyncio.sleep(60)

@app.get("/api/watchlist")
async def get_data():
    return {
        "data": sorted(MARKET_STATE.values(), key=lambda x: x['alignment_val'], reverse=True),
        "progress": f"{PROGRESS['current']}/{PROGRESS['total']}"
    }

@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
