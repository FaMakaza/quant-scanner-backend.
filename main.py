import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import yfinance as yf
import pandas as pd
import asyncio
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple


# ---------------------------
# Helpers
# ---------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if hasattr(x, "iloc"):
            if len(x) == 0:
                return default
            return float(x.iloc[0]) if len(x) == 1 else float(x.iloc[-1])
        return float(x)
    except Exception:
        return default

def round_price(symbol: str, price: float) -> float:
    s = symbol.upper()
    if s in ("XAUUSD", "XAGUSD"):
        return round(price, 2)
    if s in ("USOIL", "NATGAS"):
        return round(price, 3)
    if s in ("BTCUSD", "ETHUSD"):
        return round(price, 2)
    if s.endswith("JPY"):
        return round(price, 3)
    return round(price, 5)

def estimate_spread(symbol: str, price: float) -> Dict[str, Any]:
    s = symbol.upper()
    if len(s) == 6 and s.isalpha():
        majors = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"}
        typical_pips = 0.8 if s in majors else 1.6
        return {"value": typical_pips, "unit": "pips"}
    if s in ("XAUUSD", "XAGUSD"):
        return {"value": 0.25, "unit": "pts"}
    if s == "USOIL":
        return {"value": 0.03, "unit": "pts"}
    if s == "NATGAS":
        return {"value": 0.005, "unit": "pts"}
    if s in ("SPX", "DJI", "NASDAQ", "RUSSELL", "FTSE", "DAX", "CAC", "NIKKEI", "HSI", "DXY"):
        return {"value": 1.0, "unit": "pts"}
    if s in ("BTCUSD", "ETHUSD"):
        return {"value": max(price * 0.0004, 5.0), "unit": "pts"}
    return {"value": None, "unit": None}

def categorize(symbol: str) -> str:
    s = symbol.upper()

    fx_majors = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"}
    fx_minors = {
        "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
        "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
        "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
        "NZDJPY", "NZDCAD", "NZDCHF",
        "CADJPY", "CADCHF",
        "CHFJPY",
    }

    if s in fx_majors:
        return "FX_MAJORS"
    if s in fx_minors:
        return "FX_MINORS"
    if s in ("XAUUSD", "XAGUSD"):
        return "METALS"
    if s in ("USOIL", "NATGAS"):
        return "ENERGY"
    if s in ("SPX", "DJI", "NASDAQ", "RUSSELL", "FTSE", "DAX", "CAC", "NIKKEI", "HSI"):
        return "INDICES"
    if s == "DXY":
        return "DOLLAR"
    if s in ("BTCUSD", "ETHUSD"):
        return "CRYPTO"
    return "OTHER"


# ---------------------------
# Quant Engine
# ---------------------------

class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}

        # Display symbol -> primary ticker + optional fallbacks
        self.assets: List[Dict[str, Any]] = [
            # FX Majors
            {"symbol": "EURUSD", "tickers": ["EURUSD=X"]},
            {"symbol": "GBPUSD", "tickers": ["GBPUSD=X"]},
            {"symbol": "USDJPY", "tickers": ["USDJPY=X"]},
            {"symbol": "AUDUSD", "tickers": ["AUDUSD=X"]},
            {"symbol": "USDCAD", "tickers": ["USDCAD=X"]},
            {"symbol": "USDCHF", "tickers": ["USDCHF=X"]},
            {"symbol": "NZDUSD", "tickers": ["NZDUSD=X"]},

            # FX Minors (sample)
            {"symbol": "EURGBP", "tickers": ["EURGBP=X"]},
            {"symbol": "EURJPY", "tickers": ["EURJPY=X"]},
            {"symbol": "GBPJPY", "tickers": ["GBPJPY=X"]},
            {"symbol": "EURAUD", "tickers": ["EURAUD=X"]},
            {"symbol": "GBPAUD", "tickers": ["GBPAUD=X"]},

            # Metals: yfinance spot can be flaky on cloud; fall back to futures
            {"symbol": "XAUUSD", "tickers": ["XAUUSD=X", "GC=F"]},
            {"symbol": "XAGUSD", "tickers": ["XAGUSD=X", "SI=F"]},

            # Energy: keep your existing futures (reliable)
            {"symbol": "USOIL", "tickers": ["CL=F"]},
            {"symbol": "NATGAS", "tickers": ["NG=F"]},

            # Dollar Index
            {"symbol": "DXY", "tickers": ["DX-Y.NYB"]},

            # Indices
            {"symbol": "SPX", "tickers": ["^GSPC"]},
            {"symbol": "DJI", "tickers": ["^DJI"]},
            {"symbol": "NASDAQ", "tickers": ["^IXIC"]},
            {"symbol": "RUSSELL", "tickers": ["^RUT"]},
            {"symbol": "FTSE", "tickers": ["^FTSE"]},
            {"symbol": "DAX", "tickers": ["^GDAXI"]},
            {"symbol": "CAC", "tickers": ["^FCHI"]},
            {"symbol": "NIKKEI", "tickers": ["^N225"]},
            {"symbol": "HSI", "tickers": ["^HSI"]},

            # Crypto
            {"symbol": "BTCUSD", "tickers": ["BTC-USD"]},
            {"symbol": "ETHUSD", "tickers": ["ETH-USD"]},
        ]

    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        return yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )

    def download_with_fallbacks(self, tickers: List[str], period: str, interval: str) -> Tuple[Optional[str], pd.DataFrame]:
        last_err = None
        for t in tickers:
            try:
                df = self._download(t, period=period, interval=interval)
                if df is not None and not df.empty:
                    return t, df
            except Exception as e:
                last_err = e
                continue
        return None, pd.DataFrame()

    def calculate_bias(self, df: pd.DataFrame) -> int:
        try:
            if df is None or df.empty:
                return 0
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df["Close"].dropna()
            if close.empty:
                return 0

            ema = close.ewm(span=20, adjust=False).mean()
            curr = safe_float(close.iloc[-1])
            last_ema = safe_float(ema.iloc[-1])

            if curr > last_ema:
                return 1
            if curr < last_ema:
                return -1
            return 0
        except Exception:
            return 0

    def make_h4_series_and_levels(self, raw_h1: pd.DataFrame) -> Tuple[List[float], Dict[str, Optional[float]]]:
        if raw_h1 is None or raw_h1.empty:
            return [], {"swing_high": None, "swing_low": None, "key_level": None}

        h4 = raw_h1.resample("4h").last().dropna()
        if "Close" not in h4.columns or h4["Close"].dropna().empty:
            return [], {"swing_high": None, "swing_low": None, "key_level": None}

        close = h4["Close"].dropna()
        N = 120
        series = [safe_float(x) for x in close.tail(N).tolist()]

        M = min(len(close), 180)
        c = close.tail(M).reset_index(drop=True)

        w = 3
        swing_high = None
        swing_low = None

        for i in range(w, len(c) - w):
            window = c.iloc[i - w : i + w + 1]
            v = safe_float(c.iloc[i])
            if v == safe_float(window.max()):
                swing_high = v
            if v == safe_float(window.min()):
                swing_low = v

        if swing_high is None:
            swing_high = safe_float(c.tail(40).max())
        if swing_low is None:
            swing_low = safe_float(c.tail(40).min())

        key_level = None
        if swing_high is not None and swing_low is not None:
            key_level = (swing_high + swing_low) / 2.0

        levels = {
            "swing_high": float(swing_high) if swing_high is not None else None,
            "swing_low": float(swing_low) if swing_low is not None else None,
            "key_level": float(key_level) if key_level is not None else None,
        }
        return series, levels

    def build_setup_text(
        self,
        symbol: str,
        status: str,
        risk_tier: str,
        alignment: int,
        price: float,
        levels: Dict[str, Optional[float]],
    ) -> str:
        if status == "NEUTRAL" or alignment < 3:
            return "Waiting: need stronger multi-timeframe alignment."

        direction = "BUY" if status == "BULLISH" else "SELL"
        rr = 2.0

        sh = levels.get("swing_high")
        slw = levels.get("swing_low")
        key = levels.get("key_level")

        entry = key if key is not None else price
        buffer = max(price * 0.0015, 0.0005)

        if direction == "BUY":
            sl = (slw - buffer) if slw is not None else (entry - max(price * 0.003, 0.001))
            risk = max(entry - sl, 1e-9)
            tp = entry + risk * rr
            if sh is not None and (sh + buffer) > tp:
                tp = sh + buffer
            return (
                f"{risk_tier} {direction}: Entry near {round_price(symbol, entry)} | "
                f"SL {round_price(symbol, sl)} | TP {round_price(symbol, tp)} (~{rr:.1f}R)"
            )
        else:
            sl = (sh + buffer) if sh is not None else (entry + max(price * 0.003, 0.001))
            risk = max(sl - entry, 1e-9)
            tp = entry - risk * rr
            if slw is not None and (slw - buffer) < tp:
                tp = slw - buffer
            return (
                f"{risk_tier} {direction}: Entry near {round_price(symbol, entry)} | "
                f"SL {round_price(symbol, sl)} | TP {round_price(symbol, tp)} (~{rr:.1f}R)"
            )

    def analyze(self, display_symbol: str, tickers: List[str]) -> Optional[Dict[str, Any]]:
        try:
            used_ticker, raw = self.download_with_fallbacks(tickers, period="1y", interval="1h")
            if raw is None or raw.empty:
                return None

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

            # Price from M15 using same ticker we successfully used above
            m15_raw = pd.DataFrame()
            if used_ticker:
                try:
                    m15_raw = self._download(used_ticker, period="5d", interval="15m")
                except Exception:
                    m15_raw = pd.DataFrame()

            if m15_raw is None or m15_raw.empty or "Close" not in m15_raw.columns:
                price = safe_float(tf_data["H1"]["Close"].dropna().iloc[-1], default=0.0)
            else:
                price = safe_float(m15_raw["Close"].dropna().iloc[-1], default=0.0)

            price = round_price(display_symbol, price)

            h4_series, h4_levels = self.make_h4_series_and_levels(raw)

            risk_tier = "A+" if alignment == 4 else "A" if alignment == 3 else "B"
            signal = status if alignment >= 3 else "WAITING"

            cat = categorize(display_symbol)
            spr = estimate_spread(display_symbol, float(price))

            setup = self.build_setup_text(
                symbol=display_symbol,
                status=status,
                risk_tier=risk_tier,
                alignment=alignment,
                price=float(price),
                levels=h4_levels,
            )

            return {
                "symbol": display_symbol,
                "category": cat,
                "price": price,
                "status": status,
                "biases": biases,
                "risk_tier": risk_tier,
                "signal": signal,
                "alignment_val": alignment,
                "spread": spr,
                "setup": setup,
                "h4_series": h4_series,
                "h4_levels": h4_levels,
                "updated_at": utc_now_iso(),
            }

        except Exception:
            return None


engine = QuantEngine()

MARKET_STATE: Dict[str, Dict[str, Any]] = {}
PROGRESS: Dict[str, Any] = {"current": 0, "total": len(engine.assets), "updated_at": None}


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


async def scanner_loop():
    while True:
        PROGRESS["current"] = 0
        PROGRESS["total"] = len(engine.assets)
        PROGRESS["updated_at"] = utc_now_iso()

        for a in engine.assets:
            display = a["symbol"]
            tickers = a["tickers"]

            res = await asyncio.to_thread(engine.analyze, display, tickers)
            if res:
                MARKET_STATE[display] = res

            PROGRESS["current"] += 1
            PROGRESS["updated_at"] = utc_now_iso()
            await asyncio.sleep(0.8)

        await asyncio.sleep(60)


@app.get("/api/watchlist")
async def get_data():
    data = sorted(MARKET_STATE.values(), key=lambda x: x.get("alignment_val", 0), reverse=True)
    return {
        "data": data,
        "progress": {
            "current": PROGRESS.get("current", 0),
            "total": PROGRESS.get("total", 0),
            "updated_at": PROGRESS.get("updated_at"),
        },
    }


@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE), "updated_at": utc_now_iso()}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


