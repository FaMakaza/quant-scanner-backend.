import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import os
from datetime import datetime

# ==========================================
# 🧠 QUANT ENGINE V4.3 (PRO DISPLAY)
# ==========================================
class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}
        self.symbols = [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "GC=F", "CL=F", 
            "BTC-USD", "^GSPC", "^IXIC", "GBPJPY=X", "EURJPY=X"
        ]

    def flatten_df(self, df):
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def calculate_bias(self, df):
        try:
            df = self.flatten_df(df)
            if df is None or len(df) < 20: return 0
            close = df['Close']
            ema20 = close.ewm(span=20, adjust=False).mean()
            curr = float(close.iloc[-1])
            if curr > ema20.iloc[-1] and curr > close.iloc[-2]: return 1
            if curr < ema20.iloc[-1] and curr < close.iloc[-2]: return -1
            return 0
        except: return 0

    def analyze(self, symbol):
        try:
            # Download MTF Data
            w1 = yf.download(symbol, period="2y", interval="1wk", progress=False, auto_adjust=True)
            d1 = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=True)
            h1 = yf.download(symbol, period="1wk", interval="1h", progress=False, auto_adjust=True)
            m15 = yf.download(symbol, period="2d", interval="15m", progress=False, auto_adjust=True)
            
            biases = {
                "W1": self.calculate_bias(w1),
                "D1": self.calculate_bias(d1),
                "H4": self.calculate_bias(h1), # Proxy
                "H1": self.calculate_bias(h1)
            }
            
            score = sum(biases[tf] * self.weights[tf] for tf in self.weights)
            status = "BULLISH" if score >= 0.5 else "BEARISH" if score <= -0.5 else "NEUTRAL"
            
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for tf in biases if biases[tf] == direction) if direction != 0 else 0
            
            # Risk Tier Logic
            risk_tier = "B"
            if alignment == 4: risk_tier = "A+"
            elif alignment == 3: risk_tier = "A"

            # M15 Signal Logic
            m15 = self.flatten_df(m15)
            price = float(m15['Close'].iloc[-1]) if m15 is not None else 0
            signal = "WAITING"
            if status != "NEUTRAL" and m15 is not None:
                res = float(m15['High'].iloc[-11:-1].max())
                sup = float(m15['Low'].iloc[-11:-1].min())
                if status == "BULLISH" and price > res: signal = "BUY"
                elif status == "BEARISH" and price < sup: signal = "SELL"

            return {
                "symbol": symbol.replace("=X", "").replace("=F", "").replace("^GSPC", "SP500").replace("^IXIC", "NAS100"),
                "price": round(price, 4),
                "status": status,
                "biases": biases,
                "risk_tier": risk_tier,
                "signal": signal,
                "alignment": alignment
            }
        except: return None

# ==========================================
# 🌐 API SERVER
# ==========================================
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
            if res and (res['alignment'] >= 2):
                MARKET_STATE[symbol] = res
            await asyncio.sleep(0.5)
        await asyncio.sleep(60)

@app.get("/api/watchlist")
async def get_data():
    return sorted(MARKET_STATE.values(), key=lambda x: x['alignment'], reverse=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
