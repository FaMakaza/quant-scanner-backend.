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
# 🧠 CORE QUANT ENGINE: MTF BIAS & ALIGNMENT
# ==========================================
class QuantEngine:
    def __init__(self):
        # Specific weights from your prompt
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}

    def calculate_bias(self, df):
        """Logic: HH/HL vs LH/LL + EMA(20) Filter"""
        try:
            if df is None or len(df) < 22: return 0
            # Flatten columns for yfinance MultiIndex
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            close = df['Close']
            ema20 = close.ewm(span=20).mean()
            
            curr_close = float(close.iloc[-1])
            prev_high = float(df['High'].iloc[-2])
            prev_low = float(df['Low'].iloc[-2])
            
            # Bias Encoding: +1 Bullish, -1 Bearish, 0 Neutral
            if curr_close > ema20.iloc[-1] and curr_close > prev_high: return 1
            if curr_close < ema20.iloc[-1] and curr_close < prev_low: return -1
            return 0
        except: return 0

    def get_overall_bias(self, tf_data):
        """Formula: (W1*0.4) + (D1*0.3) + (H4*0.2) + (H1*0.1)"""
        biases = {tf: self.calculate_bias(tf_data[tf]) for tf in self.weights.keys()}
        
        score = sum(biases[tf] * self.weights[tf] for tf in self.weights)
        
        # Count Aligned Timeframes
        direction = 1 if score >= 0.5 else -1 if score <= -0.5 else 0
        alignment_score = sum(1 for tf in biases if biases[tf] == direction and direction != 0)
        
        return {
            "score": round(score, 2),
            "status": "BULLISH" if score >= 0.5 else "BEARISH" if score <= -0.5 else "NEUTRAL",
            "biases": biases,
            "alignment": alignment_score # 4/4, 3/4, etc.
        }

    def check_m15_mss(self, m15_df, overall_status):
        """M15 Market Structure Shift: Break & Close above/below Pivot"""
        try:
            if m15_df is None or len(m15_df) < 15: return "WAITING"
            if isinstance(m15_df.columns, pd.MultiIndex): m15_df.columns = m15_df.columns.get_level_values(0)
            
            # Pivot High/Low of last 10 candles
            lookback = m15_df.iloc[-11:-1]
            res = float(lookback['High'].max())
            sup = float(lookback['Low'].min())
            curr_close = float(m15_df['Close'].iloc[-1])

            if overall_status == "BULLISH" and curr_close > res: return "BUY SIGNAL"
            if overall_status == "BEARISH" and curr_close < sup: return "SELL SIGNAL"
            return "WAITING FOR MSS"
        except: return "WAITING"

# ==========================================
# 🌐 API SERVER & REAL-TIME SCANNER
# ==========================================
engine = QuantEngine()
MARKET_STATE = {}
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F", "BTC-USD"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

async def scanner_loop():
    while True:
        for symbol in SYMBOLS:
            try:
                # Fetching 5 Timeframes
                data = {
                    "W1": yf.download(symbol, period="2y", interval="1wk", progress=False),
                    "D1": yf.download(symbol, period="6mo", interval="1d", progress=False),
                    "H4": yf.download(symbol, period="1mo", interval="1h", progress=False), # Proxy for H4
                    "H1": yf.download(symbol, period="1wk", interval="1h", progress=False),
                    "M15": yf.download(symbol, period="2d", interval="15m", progress=False)
                }
                
                analysis = engine.get_overall_bias(data)
                
                # Rule: Only watchlist if Weekly & Daily agree with overall bias
                w1_bias = analysis['biases']['W1']
                d1_bias = analysis['biases']['D1']
                overall_dir = 1 if analysis['status'] == "BULLISH" else -1
                
                is_valid = (w1_bias == overall_dir and d1_bias == overall_dir and analysis['alignment'] >= 3)

                if is_valid:
                    signal = engine.check_m15_mss(data['M15'], analysis['status'])
                    MARKET_STATE[symbol] = {
                        "symbol": symbol.replace("=X", ""),
                        "price": round(float(data['M15']['Close'].iloc[-1]), 4),
                        "overall_score": analysis['score'],
                        "status": analysis['status'],
                        "alignment": f"{analysis['alignment']}/4",
                        "biases": analysis['biases'],
                        "signal": signal,
                        "rank": abs(analysis['score']) * analysis['alignment']
                    }
                else:
                    # Remove from watchlist if alignment breaks
                    MARKET_STATE.pop(symbol, None)
            except: pass
        await asyncio.sleep(60)

@app.get("/api/watchlist")
async def get_watchlist():
    # Sort by Rank (Score * Alignment)
    sorted_list = sorted(MARKET_STATE.values(), key=lambda x: x['rank'], reverse=True)
    return sorted_list

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

