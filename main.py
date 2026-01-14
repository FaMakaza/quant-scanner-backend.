# --- same imports as before ---
import os
import asyncio
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}

        # ===============================
        # SYMBOLS BY CATEGORY
        # ===============================
        self.symbols_by_category = {
            "FX_MAJORS": [
                "EURUSD=X", "GBPUSD=X", "USDJPY=X",
                "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
            ],
            "FX_MINORS": [
                "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURCAD=X", "EURAUD=X", "EURNZD=X",
                "GBPJPY=X", "GBPCHF=X", "GBPCAD=X", "GBPAUD=X", "GBPNZD=X",
                "AUDJPY=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
                "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
                "CADJPY=X", "CADCHF=X", "CHFJPY=X",
            ],
            "METALS": [
                "GC=F",      # Gold → XAUUSD
            ],
            "ENERGY": [
                "CL=F",      # USOIL
                "NG=F",      # NATGAS
            ],
            "INDICES": [
                "^GSPC", "^DJI", "^IXIC", "^RUT",
                "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
            ],
            "DOLLAR": [
                "DX-Y.NYB",  # DXY
            ],
        }

        # Flatten symbols list
        self.symbols = [s for group in self.symbols_by_category.values() for s in group]

        # Display aliases
        self.alias_map = {
            "GC=F": "XAUUSD",
            "CL=F": "USOIL",
            "NG=F": "NATGAS",
            "^GSPC": "SPX",
            "^DJI": "DJI",
            "^IXIC": "NASDAQ",
            "^RUT": "RUSSELL",
            "^FTSE": "FTSE",
            "^GDAXI": "DAX",
            "^FCHI": "CAC",
            "^N225": "NIKKEI",
            "^HSI": "HSI",
            "DX-Y.NYB": "DXY",
        }

        # Reverse lookup: symbol → category
        self.category_lookup = {
            sym: cat for cat, syms in self.symbols_by_category.items() for sym in syms
        }

    def _normalize_symbol(self, symbol: str) -> str:
        if symbol in self.alias_map:
            return self.alias_map[symbol]
        if symbol.endswith("=X"):
            return symbol.replace("=X", "")
        if symbol.endswith("=F"):
            return symbol.replace("=F", "")
        return symbol.replace("^", "")

    def _category(self, symbol: str) -> str:
        return self.category_lookup.get(symbol, "OTHER")

    @staticmethod
    def _flat_cols(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        return df

    def calculate_bias(self, df: pd.DataFrame) -> int:
        try:
            close = df["Close"].dropna()
            ema = close.ewm(span=20, adjust=False).mean()
            return 1 if close.iloc[-1] > ema.iloc[-1] else -1
        except Exception:
            return 0

    def analyze_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            raw = yf.download(symbol, period="1y", interval="1h", progress=False, auto_adjust=True)
            if raw.empty:
                return None

            raw = self._flat_cols(raw)

            tf = {
                "W1": raw.resample("W").last(),
                "D1": raw.resample("D").last(),
                "H4": raw.resample("4h").last(),
                "H1": raw,
            }

            biases = {k: self.calculate_bias(v) for k, v in tf.items()}
            score = sum(biases[k] * self.weights[k] for k in self.weights)

            status = "BULLISH" if score >= 0.2 else "BEARISH" if score <= -0.2 else "NEUTRAL"
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for v in biases.values() if v == direction)

            price = float(raw["Close"].iloc[-1])

            return {
                "symbol": self._normalize_symbol(symbol),
                "category": self._category(symbol),   # ✅ NEW
                "price": round(price, 4),
                "status": status,
                "biases": biases,
                "risk_tier": "A+" if alignment == 4 else "A" if alignment == 3 else "B",
                "signal": status if alignment >= 3 else "WAITING",
                "alignment_val": alignment,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception:
            traceback.print_exc()
            return None

    async def analyze(self, symbol: str):
        return await asyncio.to_thread(self.analyze_sync, symbol)


engine = QuantEngine()
MARKET_STATE = {}
PROGRESS = {"current": 0, "total": len(engine.symbols)}

async def scanner_loop():
    while True:
        PROGRESS["current"] = 0
        for s in engine.symbols:
            res = await engine.analyze(s)
            if res:
                MARKET_STATE[s] = res
            PROGRESS["current"] += 1
            await asyncio.sleep(0.6)
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(scanner_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/api/watchlist")
async def watchlist():
    return {
        "data": list(MARKET_STATE.values()),
        "progress": PROGRESS,
    }

@app.get("/")
async def health():
    return {"status": "online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

