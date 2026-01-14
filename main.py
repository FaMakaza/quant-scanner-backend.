import os
import asyncio
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

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
                "GC=F",   # Gold futures -> display as XAUUSD
            ],
            "ENERGY": [
                "CL=F",   # USOIL
                "NG=F",   # NATGAS
            ],
            "INDICES": [
                "^GSPC", "^DJI", "^IXIC", "^RUT",
                "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
            ],
            "DOLLAR": [
                "DX-Y.NYB",  # DXY
            ],
        }

        self.symbols = [s for group in self.symbols_by_category.values() for s in group]

        # ===============================
        # DISPLAY ALIASES
        # ===============================
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

        self.category_lookup = {sym: cat for cat, syms in self.symbols_by_category.items() for sym in syms}

        # ===============================
        # DISPLAY SPREADS (broker-like estimates)
        # FX in pips, metals/energy/indices in points
        # ===============================
        self.spread_map: Dict[str, Tuple[float, str]] = {
            # FX majors
            "EURUSD": (0.7, "pips"),
            "GBPUSD": (0.9, "pips"),
            "USDJPY": (0.8, "pips"),
            "USDCHF": (0.9, "pips"),
            "USDCAD": (1.0, "pips"),
            "AUDUSD": (0.9, "pips"),
            "NZDUSD": (1.1, "pips"),

            # FX minors/crosses
            "EURGBP": (1.2, "pips"),
            "EURJPY": (1.4, "pips"),
            "EURCHF": (1.3, "pips"),
            "EURCAD": (1.6, "pips"),
            "EURAUD": (1.8, "pips"),
            "EURNZD": (2.0, "pips"),
            "GBPJPY": (1.9, "pips"),
            "GBPCHF": (1.8, "pips"),
            "GBPCAD": (2.0, "pips"),
            "GBPAUD": (2.2, "pips"),
            "GBPNZD": (2.4, "pips"),
            "AUDJPY": (1.6, "pips"),
            "AUDCAD": (1.7, "pips"),
            "AUDCHF": (1.8, "pips"),
            "AUDNZD": (1.8, "pips"),
            "NZDJPY": (1.8, "pips"),
            "NZDCAD": (1.9, "pips"),
            "NZDCHF": (2.0, "pips"),
            "CADJPY": (1.7, "pips"),
            "CADCHF": (1.9, "pips"),
            "CHFJPY": (1.7, "pips"),

            # Metals / energy
            "XAUUSD": (20, "pts"),
            "USOIL": (3, "pts"),
            "NATGAS": (5, "pts"),

            # Indices
            "SPX": (0.8, "pts"),
            "DJI": (1.0, "pts"),
            "NASDAQ": (1.2, "pts"),
            "RUSSELL": (1.4, "pts"),
            "FTSE": (1.0, "pts"),
            "DAX": (1.5, "pts"),
            "CAC": (1.2, "pts"),
            "NIKKEI": (5.0, "pts"),
            "HSI": (8.0, "pts"),

            # Dollar index
            "DXY": (0.02, "pts"),
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

    def _spread(self, display_symbol: str) -> Dict[str, Any]:
        val, unit = self.spread_map.get(display_symbol, (None, None))
        return {"value": val, "unit": unit}

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

    @staticmethod
    def _fmt_level(x: Optional[float], digits: int) -> str:
        if x is None:
            return "--"
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "--"

    def _setup_hint(
        self,
        status: str,
        price: float,
        h4_levels: Dict[str, Optional[float]],
        digits: int,
    ) -> str:
        """
        Produces a human-readable setup hint using H4 structure.
        """
        sh = h4_levels.get("swing_high")
        sl = h4_levels.get("swing_low")
        key = h4_levels.get("key_level")

        # If no levels, fallback
        if sh is None and sl is None and key is None:
            return "Wait for pullback / breakout confirmation"

        # A small tolerance band around levels (~0.15%)
        tol = max(0.0015 * price, 0.0001)

        if status == "BULLISH":
            if sh is not None and price >= (sh - tol):
                return f"Breakout buy above Swing High {self._fmt_level(sh, digits)}"
            if key is not None and price >= key:
                return f"Buy pullback to Key {self._fmt_level(key, digits)}"
            if sl is not None:
                return f"Protect below Swing Low {self._fmt_level(sl, digits)}"
            return "Bullish: wait for pullback to structure"

        if status == "BEARISH":
            if sl is not None and price <= (sl + tol):
                return f"Breakdown sell below Swing Low {self._fmt_level(sl, digits)}"
            if key is not None and price <= key:
                return f"Sell rally to Key {self._fmt_level(key, digits)}"
            if sh is not None:
                return f"Protect above Swing High {self._fmt_level(sh, digits)}"
            return "Bearish: wait for rally into structure"

        return "Neutral: no setup"

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

            risk_tier = "A+" if alignment == 4 else "A" if alignment == 3 else "B"
            signal = status if alignment >= 3 and status != "NEUTRAL" else "WAITING"

            price = float(raw["Close"].iloc[-1])

            display_symbol = self._normalize_symbol(symbol)
            category = self._category(symbol)

            # digits
            digits = 3 if display_symbol.endswith("JPY") else 4
            if display_symbol == "XAUUSD":
                digits = 2

            # H4 overlays
            h4 = self._flat_cols(tf["H4"])
            h4_series = self._h4_series(h4, 120)
            h4_levels = self._h4_levels(h4)

            setup = self._setup_hint(status, price, h4_levels, digits)

            return {
                "symbol": display_symbol,
                "category": category,

                "price": round(price, digits),
                "status": status,
                "biases": biases,
                "risk_tier": risk_tier,
                "signal": signal,
                "alignment_val": alignment,
                "updated_at": datetime.now(timezone.utc).isoformat(),

                # ✅ new fields requested
                "spread": self._spread(display_symbol),
                "setup": setup,

                # chart overlays
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
    data = list(MARKET_STATE.values())
    return {"data": data, "progress": PROGRESS}


@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

