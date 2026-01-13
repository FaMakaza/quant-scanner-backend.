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
# 🧠 QUANTITATIVE SYSTEMS ENGINE
# ==========================================
class QuantSystemsEngine:
    def __init__(self):
        # Weights strictly from ARCHITECTURE section
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}
        self.symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "GC=F", "CL=F", "BTC-USD"]

    def flatten_df(self, df):
        """Fixes yfinance MultiIndex column issue"""
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def calculate_bias(self, df):
        """
        Calculates Bullish (+1), Bearish (-1), or Neutral (0)
        Logic: Market Structure (Price vs Prev Candle) + EMA(20)
        """
        try:
            df = self.flatten_df(df)
            if df is None or len(df) < 22: return 0
            
            close = df['Close']
            ema20 = close.ewm(span=20, adjust=False).mean()
            
            curr_close = float(close.iloc[-1])
            last_ema = float(ema20.iloc[-1])
            prev_high = float(df['High'].iloc[-2])
            prev_low = float(df['Low'].iloc[-2])
            
            # Bullish: Price > EMA and Price > Previous High
            if curr_close > last_ema and curr_close > prev_high: return 1
            # Bearish: Price < EMA and Price < Previous Low
            if curr_close < last_ema and curr_close < prev_low: return -1
            return 0
        except: return 0

    def get_weighted_analysis(self, data_dict):
        """
        Formula: (W1 * 0.4) + (D1 * 0.3) + (H4 * 0.2) + (H1 * 0.1)
        """
        # 1. Calculate individual biases
        biases = {tf: self.calculate_bias(data_dict[tf]) for tf in self.weights.keys()}
        
        # 2. Calculate Weighted Score
        score = sum(biases[tf] * self.weights[tf] for tf in self.weights)
        
        # 3. Determine Overall Status
        status = "NEUTRAL"
        if score >= 0.5: status = "BULLISH"
        elif score <= -0.5: status = "BEARISH"
        
        # 4. Alignment Calculation (Count TFs that match overall status)
        direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
        alignment_count = sum(1 for tf in biases if biases[tf] == direction) if direction != 0 else 0
        
        return {
            "score": round(score, 2),
            "status": status,
            "biases": biases,
            "alignment": alignment_count
        }

    def check_m15_signal(self, m15_df, overall_status):
        """
        Signal Engine: M15 Market Structure Shift (MSS)
        Trigger: Break & Close above/below 10-candle Pivot
        """
        try:
            m15_df = self.flatten_df(m15_df)
            if m15_df is None or len(m15_df) < 15 or overall_status == "NEUTRAL":
                return "WAITING"
            
            # Pivot High/Low of last 10 candles (excluding current)
            lookback = m15_df.iloc[-11:-1]
            pivot_high = float(lookback['High'].max())
            pivot_low = float(lookback['Low'].min())
            curr_close = float(m15_df['Close'].iloc[-1])

            if overall_status == "BULLISH" and curr_close > pivot_high:
                return "BUY SIGNAL"
            if overall_status == "BEARISH" and curr_close < pivot_low:
                return "SELL SIGNAL"
            
            return "WAITING FOR MSS"
        except: return "WAITING"

# ==========================================
# 🌐 API SERVER & BACKGROUND SCANNER
# ==========================================
engine = QuantSystemsEngine()
MARKET_STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[SYSTEM] Quantitative Engine Online. Initializing Scans...")
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

async def scanner_loop():
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] --- New Scan Cycle Starting ---")
        for symbol in engine.symbols:
            try:
                # Fetching All required timeframes
                data = {
                    "W1": yf.download(symbol, period="2y", interval="1wk", progress=False, auto_adjust=True),
                    "D1": yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=True),
                    "H4": yf.download(symbol, period="1mo", interval="1h", progress=False, auto_adjust=True), # Using H1 as proxy for H4
                    "H1": yf.download(symbol, period="1wk", interval="1h", progress=False, auto_adjust=True),
                    "M15": yf.download(symbol, period="2d", interval="15m", progress=False, auto_adjust=True)
                }
                
                analysis = engine.get_weighted_analysis(data)
                
                # --- CORE LOGIC: Watchlist Requirements ---
                # 1. Weekly & Daily must agree with overall bias
                # 2. At least 3 timeframes must align (3/4)
                overall_dir = 1 if analysis['status'] == "BULLISH" else -1
                w1_agrees = analysis['biases']['W1'] == overall_dir
                d1_agrees = analysis['biases']['D1'] == overall_dir
                alignment_ok = analysis['alignment'] >= 3

                if analysis['status'] != "NEUTRAL" and w1_agrees and d1_agrees and alignment_ok:
                    # Symbol is on watchlist -> Process M15 Signal
                    signal = engine.check_m15_signal(data['M15'], analysis['status'])
                    
                    price = float(engine.flatten_df(data['M15'])['Close'].iloc[-1])
                    
                    MARKET_STATE[symbol] = {
                        "symbol": symbol.replace("=X", ""),
                        "price": round(price, 4),
                        "overall_score": analysis['score'],
                        "status": analysis['status'],
                        "alignment": f"{analysis['alignment']}/4",
                        "biases": analysis['biases'],
                        "signal": signal,
                        "rank": abs(analysis['score']) * (analysis['alignment'] / 4)
                    }
                    print(f"✅ {symbol} added to watchlist (Rank: {MARKET_STATE[symbol]['rank']})")
                else:
                    # Remove if it no longer fits criteria
                    MARKET_STATE.pop(symbol, None)
                    
            except Exception as e:
                print(f"❌ Error Scanning {symbol}: {e}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] --- Scan Complete. Watchlist Size: {len(MARKET_STATE)} ---")
        await asyncio.sleep(60)

@app.get("/api/watchlist")
async def get_watchlist():
    # Final Output Requirement: Rank instruments by alignment/score
    sorted_watchlist = sorted(MARKET_STATE.values(), key=lambda x: x['rank'], reverse=True)
    return sorted_watchlist

@app.get("/")
async def health():
    return {"status": "Quant Engine Online", "active_pairs": len(MARKET_STATE)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
