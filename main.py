import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yfinance as yf
import pandas as pd
import asyncio
import os
from datetime import datetime

# Logic to handle yfinance MultiIndex columns
def flatten_df(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

MARKET_STATE = {}
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F", "NG=F"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scan_markets())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

async def scan_markets():
    while True:
        for symbol in SYMBOLS:
            try:
                # Fetch M15 data
                df = yf.download(symbol, period="2d", interval="15m", progress=False, auto_adjust=True)
                df = flatten_df(df)
                
                if df is not None:
                    price = float(df['Close'].iloc[-1])
                    # Logic: Simple EMA Bias
                    ema = df['Close'].ewm(span=20).mean().iloc[-1]
                    bias = "BULLISH" if price > ema else "BEARISH"
                    
                    MARKET_STATE[symbol] = {
                        "symbol": symbol.replace("=X", "").replace("=F", " (OIL/GOLD)"),
                        "price": round(price, 4),
                        "bias": bias,
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
            except Exception as e:
                print(f"Error {symbol}: {e}")
        await asyncio.sleep(60)

@app.get("/api/watchlist")
async def get_data():
    return list(MARKET_STATE.values())

@app.get("/")
async def health():
    return {"status": "Scanner Online"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
