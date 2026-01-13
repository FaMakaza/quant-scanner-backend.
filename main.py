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
        # Start with a SMALL list to make it fast
        self.symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "BTC-USD"]

    def calculate_bias(self, df):
        try:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            close = df['Close']
            ema = close.ewm(span=20, adjust=False).mean()
            if float(close.iloc[-1]) > float(ema.iloc[-1]): return 1
            if float(close.iloc[-1]) < float(ema.iloc[-1]): return -1
            return 0
        except: return 0

    def analyze(self, symbol):
        try:
            # Shortened periods for faster initial load
            w1 = yf.download(symbol, period="1y", interval="1wk", progress=False, auto_adjust=True)
            d1 = yf.download(symbol, period="3mo", interval="1d", progress=False, auto_adjust=True)
            h1 = yf.download(symbol, period="1wk", interval="1h", progress=False, auto_adjust=True)
            m15 = yf.download(symbol, period="2d", interval="15m", progress=False, auto_adjust=True)
            
            biases = {"W1": self.calculate_bias(w1), "D1": self.calculate_bias(d1), 
                      "H4": self.calculate_bias(h1), "H1": self.calculate_bias(h1)}
            
            score = sum(biases[tf] * self.weights[tf] for tf in self.weights)
            status = "BULLISH" if score >= 0.2 else "BEARISH" if score <= -0.2 else "NEUTRAL"
            
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for tf in biases if biases[tf] == direction) if direction != 0 else 0
            
            price = float(m15['Close'].iloc[-1]) if not m15.empty else 0
            
            return {
                "symbol": symbol.replace("=X", "").replace("=F", ""),
                "price": round(price, 4),
                "status": status,
                "biases": biases,
                "risk_tier": "A+" if alignment == 4 else "A" if alignment == 3 else "B",
                "signal": "WAITING",
                "alignment": alignment
            }
        except Exception as e:
            print(f"Error {symbol}: {e}")
            return None

engine = QuantEngine()
MARKET_STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scanner())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

async def scanner():
    while True:
        for symbol in engine.symbols:
            res = engine.analyze(symbol)
            if res: MARKET_STATE[symbol] = res
            await asyncio.sleep(1) # Fast scan
        await asyncio.sleep(30)

@app.get("/api/watchlist")
async def get_data():
    return list(MARKET_STATE.values())

@app.get("/")
async def root():
    return {"status": "online", "count": len(MARKET_STATE)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

