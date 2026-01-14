import os
import asyncio
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}
        # Focused list to keep scans fast and avoid rate limits
        self.symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "GC=F", "CL=F", "BTC-USD", "SPY"]

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.replace("=X", "").replace("=F", "")

    def calculate_bias(self, df: pd.DataFrame) -> int:
        try:
            # yfinance can return MultiIndex columns in some cases
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df["Close"].dropna()
            if close.empty:
                return 0

            ema = close.ewm(span=20, adjust=False).mean()
            curr = float(close.iloc[-1])
            last_ema = float(ema.iloc[-1])

            if curr > last_ema:
                return 1
            if curr < last_ema:
                return -1
            return 0
        except Exception:
            return 0

    def analyze_sync(self, symbol: str):
        """
        Synchronous analysis (we'll run this in a thread via asyncio.to_thread)
        """
        try:
            raw = yf.download(
                symbol,
                period="1y",
                interval="1h",
                progress=False,
                auto_adjust=True,
                threads=False,  # reduce thread contention in some deploy envs
            )

            if raw is None or raw.empty:
                return None

            # Resample to get MTF data (use 'h' instead of deprecated 'H')
            tf_data = {
                "W1": raw.resample("W").last(),
                "D1": raw.resample("D").last(),
                "H4": raw.resample("4h").last(),
                "H1": raw,
            }

            biases = {tf: self.calculate_bias(tf_data[tf]) for tf in self.weights.keys()}
            score = sum(biases[tf] * self.weights[tf] for tf in self.weights)

            status = "BULLISH" if score >= 0.2 else "BEARISH" if score <= -0.2 else "NEUTRAL"
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for v in biases.values() if v == direction) if direction != 0 else 0

            # M15 Entry Logic
            m15_raw = yf.download(
                symbol,
                period="1d",
                interval="15m",
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            price = 0.0
            if m15_raw is not None and not m15_raw.empty:
                # Make sure this is a scalar
                price = float(m15_raw["Close"].iloc[-1])

            risk_tier = "A+" if alignment == 4 else "A" if alignment == 3 else "B"
            signal = status if alignment >= 3 else "WAITING"

            return {
                "symbol": self._normalize_symbol(symbol),
                "price": round(price, 4),
                "status": status,
                "biases": biases,
                "risk_tier": risk_tier,
                "signal": signal,
                "alignment_val": alignment,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            print(f"[analyze_sync] Error for {symbol}: {e}")
            traceback.print_exc()
            return None

    async def analyze(self, symbol: str):
        # Run yfinance work off the event loop so it doesn't block FastAPI
        return await asyncio.to_thread(self.analyze_sync, symbol)


engine = QuantEngine()
MARKET_STATE = {}
PROGRESS = {"current": 0, "total": len(engine.symbols), "updated_at": None}


async def scanner_loop():
    while True:
        PROGRESS["current"] = 0
        PROGRESS["total"] = len(engine.symbols)
        PROGRESS["updated_at"] = datetime.now(timezone.utc).isoformat()

        for symbol in engine.symbols:
            res = await engine.analyze(symbol)
            if res is not None:
                MARKET_STATE[symbol] = res

            PROGRESS["current"] += 1
            PROGRESS["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Gentle pacing (adjust if you hit rate limits)
            await asyncio.sleep(0.8)

        # Wait before next full scan
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

# CORS (open while developing; lock down in production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/watchlist")
async def get_watchlist():
    # Return progress as NUMERIC object (so frontend doesn't show 0/0 due to parsing)
    data = sorted(MARKET_STATE.values(), key=lambda x: x.get("alignment_val", 0), reverse=True)
    return {
        "data": data,
        "progress": PROGRESS,  # {"current": x, "total": y, "updated_at": "..."}
        "active": len(data),
    }


@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE), "progress": PROGRESS}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


