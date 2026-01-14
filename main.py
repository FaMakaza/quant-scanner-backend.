import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yfinance as yf
import pandas as pd
import asyncio
import os
from datetime import datetime, timezone
import traceback


class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}

        # ---------------------------
        # Yahoo Finance tickers (INPUT)
        # ---------------------------
        # FX via yfinance uses "XXXXXX=X" (e.g., "EURUSD=X")
        # Metals spot often available as "XAUUSD=X"
        # USOIL via futures: "CL=F"
        # Natural Gas futures: "NG=F"
        self.symbols = [
            # --- FX Majors ---
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",

            # --- FX Minors / Crosses (popular) ---
            "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURCAD=X", "EURAUD=X", "EURNZD=X",
            "GBPJPY=X", "GBPCHF=X", "GBPCAD=X", "GBPAUD=X", "GBPNZD=X",
            "AUDJPY=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
            "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
            "CADJPY=X", "CADCHF=X", "CHFJPY=X",

            # --- Metals (spot) ---
            "XAUUSD=X",

            # --- Energy (futures used; we return TradingView-style names) ---
            "CL=F",   # USOIL
            "NG=F",   # NATGAS
        ]

        # ---------------------------
        # Returned TradingView-style symbol overrides (OUTPUT)
        # ---------------------------
        self.alias_map = {
            "CL=F": "USOIL",
            "NG=F": "NATGAS",
            "XAUUSD=X": "XAUUSD",
        }

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Convert yfinance ticker → TradingView-style display symbol.
        Examples:
          EURUSD=X → EURUSD
          XAUUSD=X → XAUUSD
          CL=F     → USOIL
          NG=F     → NATGAS
        """
        if symbol in self.alias_map:
            return self.alias_map[symbol]

        # FX format
        if symbol.endswith("=X"):
            return symbol.replace("=X", "")

        # Futures format
        if symbol.endswith("=F"):
            return symbol.replace("=F", "")

        return symbol

    def calculate_bias(self, df: pd.DataFrame) -> int:
        try:
            if df is None or df.empty:
                return 0

            # yfinance can sometimes return MultiIndex columns
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
        Sync analysis (run in a thread using asyncio.to_thread).
        """
        try:
            raw = yf.download(
                symbol,
                period="1y",
                interval="1h",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if raw is None or raw.empty:
                return None

            # Resample timeframes (use 'h' not 'H')
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

            # M15 entry logic (more responsive price)
            m15_raw = yf.download(
                symbol,
                period="5d",     # more robust than 1d for some tickers
                interval="15m",
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            price = 0.0
            if m15_raw is not None and not m15_raw.empty:
                if isinstance(m15_raw.columns, pd.MultiIndex):
                    m15_raw.columns = m15_raw.columns.get_level_values(0)
                close = m15_raw["Close"].dropna()
                if not close.empty:
                    price = float(close.iloc[-1])

            tv_symbol = self._normalize_symbol(symbol)

            # Risk tier based on alignment
            risk_tier = "A+" if alignment == 4 else "A" if alignment == 3 else "B"

            # Signal behavior
            signal = status if alignment >= 3 and status != "NEUTRAL" else "WAITING"

            return {
                "symbol": tv_symbol,  # ✅ TradingView-style symbol sent to app
                "price": round(price, 5) if tv_symbol.endswith("JPY") else round(price, 4),
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

            # Gentle pacing to avoid yfinance throttles
            await asyncio.sleep(0.6)

        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/watchlist")
async def get_watchlist():
    data = sorted(MARKET_STATE.values(), key=lambda x: x.get("alignment_val", 0), reverse=True)
    return {
        "data": data,
        "progress": PROGRESS,   # object {current,total,...}
        "active": len(data),
    }


@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE), "progress": PROGRESS}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



