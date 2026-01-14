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
            # FX Majors
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",

            # FX Minors / Crosses
            "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURCAD=X", "EURAUD=X", "EURNZD=X",
            "GBPJPY=X", "GBPCHF=X", "GBPCAD=X", "GBPAUD=X", "GBPNZD=X",
            "AUDJPY=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
            "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
            "CADJPY=X", "CADCHF=X", "CHFJPY=X",

            # Metals (use reliable futures feed, display as XAUUSD)
            "GC=F",     # Gold futures → display as XAUUSD

            # Energy
            "CL=F",     # USOIL
            "NG=F",     # NATGAS

            # Indices
            "^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",

            # Dollar Index
            "DX-Y.NYB",
        ]

        # ===============================
        # DISPLAY ALIASES
        # ===============================
        self.alias_map = {
            "CL=F": "USOIL",
            "NG=F": "NATGAS",

            # ✅ Gold: show as XAUUSD in the app
            "GC=F": "XAUUSD",

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
            if close.empty:
                return 0
            ema = close.ewm(span=20, adjust=False).mean()
            return 1 if float(close.iloc[-1]) > float(ema.iloc[-1]) else -1
        except Exception:
            return 0

    @staticmethod
    def _change_24h_pct(h1: pd.DataFrame) -> float:
        try:
            close = h1["Close"].dropna()
            if len(close) < 25:
                return 0.0
            last = float(close.iloc[-1])
            prev = float(close.iloc[-25])
            if prev == 0:
                return 0.0
            return (last - prev) / prev * 100.0
        except Exception:
            return 0.0

    @staticmethod
    def _sparkline(h1: pd.DataFrame, points: int = 24) -> List[float]:
        try:
            close = h1["Close"].dropna().tail(points)
            return [float(x) for x in close.values]
        except Exception:
            return []

    @staticmethod
    def _atr_pct(h1: pd.DataFrame, period: int = 14) -> float:
        try:
            high, low, close = h1["High"].dropna(), h1["Low"].dropna(), h1["Close"].dropna()
            if len(close) < period + 2:
                return 0.0
            prev_close = close.shift(1)
            tr = pd.concat(
                [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            last = close.iloc[-1]
            if pd.isna(atr) or pd.isna(last) or float(last) == 0:
                return 0.0
            return float(atr) / float(last) * 100.0
        except Exception:
            return 0.0

    @staticmethod
    def _vol_tier(atr_pct: float) -> str:
        if atr_pct < 0.20:
            return "LOW"
        if atr_pct < 0.50:
            return "MED"
        return "HIGH"

    @staticmethod
    def _h4_series(df_h4: pd.DataFrame, points: int = 120) -> List[float]:
        try:
            close = df_h4["Close"].dropna().tail(points)
            return [float(x) for x in close.values]
        except Exception:
            return []

    @staticmethod
    def _h4_levels(df_h4: pd.DataFrame) -> Dict[str, Optional[float]]:
        out = {"swing_high": None, "swing_low": None, "key_level": None}
        try:
            if df_h4 is None or df_h4.empty or len(df_h4) < 10:
                return out

            df = df_h4.tail(180).copy()
            highs = df["High"].values
            lows = df["Low"].values

            swing_highs: List[float] = []
            swing_lows: List[float] = []

            for i in range(1, len(df) - 1):
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                    swing_highs.append(float(highs[i]))
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    swing_lows.append(float(lows[i]))

            last_high = swing_highs[-1] if swing_highs else None
            last_low = swing_lows[-1] if swing_lows else None

            out["swing_high"] = last_high
            out["swing_low"] = last_low
            out["key_level"] = last_low if last_low is not None else last_high
            return out
        except Exception:
            return out

    def analyze_sync(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            raw = yf.download(symbol, period="1y", interval="1h", progress=False, auto_adjust=True, threads=False)
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
            direction = 1 if status == "BULLISH" else -1 if status == "BEARISH" else 0
            alignment = sum(1 for v in biases.values() if v == direction) if direction != 0 else 0

            price = float(raw["Close"].iloc[-1])
            change_24h = self._change_24h_pct(tf["H1"])
            spark = self._sparkline(tf["H1"])
            atr_pct = self._atr_pct(tf["H1"])
            vol_tier = self._vol_tier(atr_pct)

            risk_tier = "A+" if alignment == 4 else "A" if alignment == 3 else "B"
            signal = status if alignment >= 3 and status != "NEUTRAL" else "WAITING"

            tv_symbol = self._normalize_symbol(symbol)

            h4 = self._flat_cols(tf["H4"])
            h4_series = self._h4_series(h4, 120)
            h4_levels = self._h4_levels(h4)

            digits = 3 if tv_symbol.endswith("JPY") else 4
            # Gold futures price usually has 2 decimals; but we keep 2–4 safe
            if tv_symbol == "XAUUSD":
                digits = 2

            return {
                "symbol": tv_symbol,
                "price": round(price, digits),
                "status": status,
                "biases": biases,
                "risk_tier": risk_tier,
                "signal": signal,
                "alignment_val": alignment,
                "updated_at": datetime.now(timezone.utc).isoformat(),

                "change_24h_pct": round(float(change_24h), 2),
                "atr_pct": round(float(atr_pct), 3),
                "volatility_tier": vol_tier,
                "sparkline": spark,

                "h4_series": h4_series,
                "h4_levels": h4_levels,
            }

        except Exception as e:
            print(f"[analyze_sync] Error for {symbol}: {e}")
            traceback.print_exc()
            return None

    async def analyze(self, symbol: str):
        return await asyncio.to_thread(self.analyze_sync, symbol)


engine = QuantEngine()
MARKET_STATE: Dict[str, Any] = {}
PROGRESS = {"current": 0, "total": len(engine.symbols), "updated_at": None}


async def scanner_loop():
    while True:
        PROGRESS["current"] = 0
        PROGRESS["total"] = len(engine.symbols)
        PROGRESS["updated_at"] = datetime.now(timezone.utc).isoformat()

        for s in engine.symbols:
            res = await engine.analyze(s)
            if res:
                MARKET_STATE[s] = res
            PROGRESS["current"] += 1
            PROGRESS["updated_at"] = datetime.now(timezone.utc).isoformat()
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
    data = sorted(MARKET_STATE.values(), key=lambda x: x.get("alignment_val", 0), reverse=True)
    return {"data": data, "progress": PROGRESS}


@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
