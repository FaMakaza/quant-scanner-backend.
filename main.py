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
# 🧠 QUANT ENGINE: GLOBAL ASSET SCANNER
# ==========================================
class QuantEngineV4:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}
        
        # --- FULL MARKET UNIVERSE ---
        self.forex_majors = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"]
        self.forex_minors = ["EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURAUD=X", "GBPCHF=X", "CADJPY=X"]
        self.indices = ["SPY", "QQQ", "DIA", "IWM", "^GSPC", "^IXIC", "^DJI"]
        self.commodities = ["GC=F", "SI=F", "CL=F", "NG=F", "BZ=F", "HG=F"] # Gold, Silver, Crude, NatGas, Brent, Copper
        
        self.symbols = list(set(self.forex_majors + self.forex_minors + self.indices + self.commodities))

    def flatten_df(self, df):
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def calculate_bias(self, df):
        """EMA20 + Price Action Bias Detection"""
        try:
            df = self.flatten_df(df)
            if df is None or len(df) < 20: return 0
            
            close = df['Close']
            ema20 = close.ewm(span=20, adjust=False).mean()
            
            curr_close = float(close.iloc[-1])
            last_ema = float(ema20.iloc[-1])
            
            # Trend Logic
            if curr_close > last_ema:
                return 1 if curr_close > close.iloc[-2] else 0
            elif curr_close < last_ema:
                return -1 if curr_close < close.iloc[-2] else 0
            return 0
        except: return 0

    def analyze_asset(self, symbol):
        """Fetches and analyzes all timeframes for a single symbol"""
        try:
            # Download all required data in one go where possible
            # W1: 2y, D1: 6mo, H1/H4: 1mo, M15: 2d
            w1 = yf.download(symbol, period="2y", interval="1wk", progress=False, auto_adjust=True)
            d1 = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=True)
            h1 = yf.download(symbol, period="1wk", interval="1h", progress=False, auto_adjust=True)
            m15 = yf.download(symbol, period="2d", interval="15m", progress=False, auto_adjust=True)
            
            tf_data = {"W1": w1, "D1": d1, "H4": h1, "H1": h1} # Using H1 as proxy for H4
            
            biases = {tf: self.calculate_bias(tf_data[tf]) for tf in self.weights.keys()}
            score = sum(biases[tf] * self.weights[tf] for tf in self.weights)
            
            status = "NEUTRAL"
            if score >= 0.5: status = "BULLISH"
            elif score <= -0.5: status = "BEARISH"
            
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for tf in biases if biases[tf] == direction) if direction != 0 else 0
            
            # --- M15 Signal Logic ---
            m15 = self.flatten_df(m15)
            signal = "WAITING"
            price = 0.0
            if m15 is not None and not m15.empty:
                price = float(m15['Close'].iloc[-1])
                if status != "NEUTRAL":
                    lookback = m15.iloc[-11:-1]
                    pivot_h = float(lookback['High'].max())
                    pivot_l = float(lookback['Low'].min())
                    if status == "BULLISH" and price > pivot_h: signal = "BUY SIGNAL"
                    elif status == "BEARISH" and price < pivot_l: signal = "SELL SIGNAL"
                    else: signal = "AWAITING MSS"

            return {
                "symbol": symbol.replace("=X", "").replace("=F", "").replace("^", ""),
                "price": round(price, 4),
                "status": status,
                "score": round(score, 2),
                "alignment": f"{alignment}/4",
                "biases": biases,
                "signal": signal,
                "is_active": alignment >= 2 # Filter to keep data moving
            }
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

# ==========================================
# 🌐 SERVER & PARALLEL SCANNER
# ==========================================
engine = QuantEngineV4()
MARKET_STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n[SYSTEM] Quant Engine Online. Monitoring {len(engine.symbols)} Assets.")
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

async def scanner_loop():
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Starting Global Scan...")
        
        # Scan symbols one by one to avoid Yahoo Finance rate limiting
        for symbol in engine.symbols:
            result = engine.analyze_asset(symbol)
            if result and result['is_active']:
                MARKET_STATE[symbol] = result
                print(f"   ✅ {result['symbol']} Aligned: {result['alignment']}")
            else:
                MARKET_STATE.pop(symbol, None)
            
            # Small delay to respect API limits
            await asyncio.sleep(0.5)
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 😴 Global Scan Complete. Waiting 2 minutes.")
        await asyncio.sleep(120) 

@app.get("/api/watchlist")
async def get_watchlist():
    # Sort by Score (Magnitude of trend)
    return sorted(MARKET_STATE.values(), key=lambda x: abs(x['score']), reverse=True)

@app.get("/")
async def health():
    return {"status": "Quant Engine Online", "monitored": len(engine.symbols), "aligned": len(MARKET_STATE)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
