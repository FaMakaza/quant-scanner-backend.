import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yfinance as yf
import asyncio
import os
from datetime import datetime

MARKET_STATE = {}
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F"]

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
                # Fetch M15 and D1
                df = yf.download(symbol, period="2d", interval="15m", progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                price = float(df['Close'].iloc[-1])
                MARKET_STATE[symbol] = {
                    "symbol": symbol.replace("=X", ""),
                    "price": round(price, 4),
                    "signal": "WAITING", # Add your logic here
                    "time": datetime.now().isoformat()
                }
            except: pass
        await asyncio.sleep(60)

@app.get("/api/watchlist")
async def get_data():
    return list(MARKET_STATE.values())

if __name__ == "__main__":
    # Railway/Cloud servers provide a PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)