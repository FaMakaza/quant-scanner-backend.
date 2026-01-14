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
        # SYMBOL UNIVERSE
        # ===============================
        self.symbols = [
            # -------- FX MAJORS --------
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
            "USDCAD=X", "AUDUSD=X", "NZDUSD=X",

            # -------- FX MINORS / CROSSES --------
            "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURCAD=X",
            "EURAUD=X", "EURNZD=X",
            "GBPJPY=X", "GBPCHF=X", "GBPCAD=X",
            "GBPAUD=X", "GBPNZD=X",
            "AUDJPY=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
            "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
            "CADJPY=X", "CADCHF=X", "CHFJPY=X",

            # -------- METALS --------
            "XAUUSD=X",

            # -------- ENERGY --------
            "CL=F",      # USOIL
            "NG=F",      # NATGAS

            # -------- INDICES --------
            "^GSPC",     # S&P 500
            "^DJI",      # Dow Jones
            "^IXIC",     # Nasdaq
            "^RUT",      # Russell 2000
            "^FTSE",     # FTSE 100
            "^GDAXI",    # DAX
            "^FCHI",     # CAC 40
            "^N225",     # Nikkei 225
            "^HSI",      # Hang Seng

            # -------- DOLLAR INDEX --------
            "DX-Y.NYB",  # DXY
        ]

        # ===============================
        # SYMBOL ALIASES (display + TradingView friendly)
        # ===============================
        self.alias_map = {
            "CL=F": "USOIL",
            "NG=F": "NATGAS",
            "XAUUSD=X": "XAUUSD",

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

        # ===============================
        # DISPLAY SPREADS (BROKER-LIKE)
        # ===============================
        self.spread_map = {
            # FX (pips)
            "EURUSD": (0.7, "pips"),
            "GBPUSD": (0.9, "pips"),
            "USDJPY": (0.8, "pips"),
            "USDCAD": (1.0, "pips"),
            "AUDUSD": (0.9, "pips"),
            "NZDUSD": (1.1, "pips"),

            # Metals / Energy (points)
            "XAUUSD": (20, "pts"),
            "USOIL": (3, "pts"),
            "NATGAS": (5, "pts"),

            # Indices (points)
            "SPX": (0.8, "pts"),
            "DJI": (1.0, "pts"),
            "NASDAQ": (1.2, "pts"),
            "RUSSELL": (1.4, "pts"),
            "DAX": (1.5, "pts"),
            "FTSE": (1.0, "pts"),
            "CAC": (1.2, "pts"),
            "NIKKEI": (5.0, "pts"),
            "HSI": (8.0, "pts"),

            # Dollar Index
            "DXY": (0.02, "pts"),
        }

    # ===============================
    # HELPERS
    # ===============================
    def _normalize_symbol(self, symbol: str) -> str:
        if symbol in self.alias_map:
            return self.alias_map[symbol]
        if symbol.endswith("=X"):
            return symbol.replace("=X", "")
        if symbol.endswith("=F"):
            return symbol.replace("=F", "")
        return symbol.replace("^", "")

    @staticmethod
    def _flat_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        return df

    def calculate_bias(self, df: pd.DataFrame) -> int:
        try:
            if df is None or df.empty:
                return 0
            close = df["Close"].dropna()
            ema = close.ewm(span=20, adjust=False).mean()
            return 1 if close.iloc[-1] > ema.iloc[-1] else -1
        except Exception:
            return 0

    @staticmethod
    def _change_24h_pct(h1: pd.DataFrame) -> float:
        try:
            close = h1["Close"].dropna()
            if len(close) < 25:
                return 0.0
            return (close.iloc[-1] - close.iloc[-25]) / close.iloc[-25] * 100
        except Exception:
            return 0.0

    @staticmethod
    def _sparkline(h1: pd.DataFrame, points: int = 24) -> List[float]:
        try:
            return [float(x) for x in h1["Close"].dropna().tail(points)]
        except Exception:
            return []

    @staticmethod
    def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
        try:
            high, low, close = df["High"], df["Low"], df["Close"]
            tr = pd.concat(
                [(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return float(atr / close.iloc[-1] * 100)
        except Exception:
            return 0.0

    def _vol_tier(self, atr_pct: float) -> str:
        if atr_pct < 0.2:
            return "LOW"
        if atr_pct < 0.5:
            return "MED"
        return "HIGH"

    def _spread(self, sym: str) -> Dict[str, Any]:
        val, unit = self.spread_map.get(sym, (None, None))
        return {"value": val, "unit": unit}

    # ===============================
    # ANALYSIS
    # ===============================
    def analyze_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            raw = yf.download(symbol, period="1y", interval="1h", progress=False, auto_adjust=True)
            if raw is None or raw.empty:
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
            direction = 1 if status == "BULLISH" else -1
            alignment = sum(1 for v in biases.values() if v == direction)

            price = float(raw["Close"].iloc[-1])
            change_24h = self._change_24h_pct(tf["H1"])
            spark = self._sparkline(tf["H1"])

            atr_pct = self._atr_pct(tf["H1"])
            vol_tier = self._vol_tier(atr_pct)

            risk_tier = "A+" if alignment == 4 else "A" if alignment == 3 else "B"
            signal = status if alignment >= 3 else "WAITING"

            tv_symbol = self._normalize_symbol(symbol)

            return {
                "symbol": tv_symbol,
                "price": round(price, 4),
                "status": status,
                "biases": biases,
                "risk_tier": risk_tier,
                "signal": signal,
                "alignment_val": alignment,
                "change_24h_pct": round(change_24h, 2),
                "spread": self._spread(tv_symbol),
                "atr_pct": round(atr_pct, 3),
                "volatility_tier": vol_tier,
                "sparkline": spark,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(symbol, e)
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/watchlist")
async def watchlist():
    data = sorted(MARKET_STATE.values(), key=lambda x: x["alignment_val"], reverse=True)
    return {"data": data, "progress": PROGRESS}


@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))





