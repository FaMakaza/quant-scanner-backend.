import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple


# ---------------------------
# Helpers
# ---------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_iso() -> str:
    return utc_now().isoformat()

def safe_float(x, default=0.0) -> float:
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        if hasattr(x, "iloc"):
            return float(x.iloc[-1]) if len(x) >= 1 else default
        return float(x)
    except:
        return default

def round_price(symbol: str, price: float) -> float:
    s = symbol.upper()
    if any(k in s for k in ("XAU", "XAG", "BTC", "ETH", "SPX", "DJI", "NASDAQ", "DXY")):
        return round(price, 2)
    if any(k in s for k in ("OIL", "GAS", "JPY")):
        return round(price, 3)
    return round(price, 5)

def estimate_spread(symbol: str, price: float) -> Dict[str, Any]:
    s = symbol.upper()
    if len(s) == 6 and s.isalpha():
        majors = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"}
        return {"value": 0.8 if s in majors else 1.6, "unit": "pips"}
    return {"value": 0.02, "unit": "pts"}

# ---------------------------
# Quant Engine
# ---------------------------

class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}
        self.assets = [
            {"symbol": "EURUSD", "ticker": "EURUSD=X"}, {"symbol": "GBPUSD", "ticker": "GBPUSD=X"},
            {"symbol": "USDJPY", "ticker": "USDJPY=X"}, {"symbol": "AUDUSD", "ticker": "AUDUSD=X"},
            {"symbol": "XAUUSD", "ticker": "XAUUSD=X"}, {"symbol": "BTCUSD", "ticker": "BTC-USD"},
            {"symbol": "SPX", "ticker": "^GSPC"}, {"symbol": "DXY", "ticker": "DX-Y.NYB"}
        ]

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates Average True Range for SL buffering."""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def calculate_bias(self, df: pd.DataFrame) -> int:
        if df.empty: return 0
        close = df["Close"]
        ema = close.ewm(span=20, adjust=False).mean()
        curr, last_ema = safe_float(close.iloc[-1]), safe_float(ema.iloc[-1])
        return 1 if curr > last_ema else -1 if curr < last_ema else 0

    def make_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Identifies recent structural highs and lows."""
        recent = df.tail(50)
        return {
            "swing_high": float(recent["High"].max()),
            "swing_low": float(recent["Low"].min()),
            "key_level": float(recent["Close"].mean())
        }

    def build_trade_setup(self, symbol: str, status: str, alignment: int, price: float, df: pd.DataFrame) -> Dict[str, Any]:
        """Creates structured Trade Idea (A+ only)."""
        if alignment < 4:
            return {"ready": False, "reason": "Waiting for 4/4 alignment."}
        
        levels = self.make_levels(df)
        atr = self.calculate_atr(df)
        
        if status == "BULLISH":
            entry = price
            sl = levels["swing_low"] - (atr * 0.5) # Buffer SL below structure
            tp = levels["swing_high"]
        else:
            entry = price
            sl = levels["swing_high"] + (atr * 0.5) # Buffer SL above structure
            tp = levels["swing_low"]

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

        return {
            "ready": True,
            "entry": round_price(symbol, entry),
            "sl": round_price(symbol, sl),
            "tp": round_price(symbol, tp),
            "rr": rr,
            "bias_text": f"Confirmed {status} momentum with structure-based protection."
        }

    def analyze(self, display_symbol: str, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            raw = yf.download(ticker, period="1y", interval="1h", progress=False)
            if raw.empty: return None
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)

            biases = {
                "W1": self.calculate_bias(raw.resample("W").last()),
                "D1": self.calculate_bias(raw.resample("D").last()),
                "H4": self.calculate_bias(raw.resample("4h").last()),
                "H1": self.calculate_bias(raw)
            }
            
            score = sum(biases[tf] * self.weights[tf] for tf in self.weights)
            status = "BULLISH" if score >= 0.2 else "BEARISH" if score <= -0.2 else "NEUTRAL"
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for v in biases.values() if v == direction) if direction != 0 else 0
            
            price = safe_float(raw["Close"].iloc[-1])
            setup = self.build_trade_setup(display_symbol, status, alignment, price, raw)

            return {
                "symbol": display_symbol,
                "price": round_price(display_symbol, price),
                "status": status,
                "biases": biases,
                "alignment_val": alignment,
                "risk_tier": "A+" if alignment == 4 else "A" if alignment == 3 else "B",
                "setup": setup,
                "updated_at": utc_now_iso()
            }
        except Exception as e:
            print(f"Error analyzing {display_symbol}: {e}")
            return None

# ---------------------------
# App State & Loops
# ---------------------------

engine = QuantEngine()
MARKET_STATE = {}

async def scanner_loop():
    while True:
        for a in engine.assets:
            res = await asyncio.to_thread(engine.analyze, a["symbol"], a["ticker"])
            if res: MARKET_STATE[a["symbol"]] = res
            await asyncio.sleep(1)
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/watchlist")
async def get_watchlist():
    return {"data": sorted(MARKET_STATE.values(), key=lambda x: x.get("alignment_val", 0), reverse=True)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))








