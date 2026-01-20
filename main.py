import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import yfinance as yf
import pandas as pd
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
    if s in ("SPX", "DJI", "NASDAQ", "RUSSELL", "FTSE", "DAX", "CAC", "NIKKEI", "HSI", "DXY"):
        return round(price, 2)
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
    return {"value": 0.02, "unit": "pts"}


# ---------------------------
# Sessions (UTC-based)
# ---------------------------

SESSIONS = {
    "ASIA":  {"name": "Asia",     "open_h": 0,  "close_h": 9},
    "LONDON":{"name": "London",   "open_h": 7,  "close_h": 16},
    "NY":    {"name": "New York", "open_h": 13, "close_h": 22},
}

def next_session_info(now: datetime) -> Dict[str, Any]:
    minutes_now = now.hour * 60 + now.minute
    best = None

    for key, s in SESSIONS.items():
        open_min = s["open_h"] * 60
        delta = open_min - minutes_now
        if delta <= 0:
            delta += 24 * 60
        opens_at = now + timedelta(minutes=delta)
        if best is None or delta < best["opens_in_min"]:
            best = {
                "key": key,
                "name": s["name"],
                "opens_in_min": delta,
                "opens_at": opens_at.isoformat(),
            }
    return best


# ---------------------------
# Quant Engine
# ---------------------------

class QuantEngine:
    def __init__(self):
        self.weights = {"W1": 0.4, "D1": 0.3, "H4": 0.2, "H1": 0.1}

        # ✅ ticker can be a string OR a list of fallbacks
        self.assets: List[Dict[str, Any]] = [
            # Currencies
            {"symbol": "EURUSD", "tickers": ["EURUSD=X"]},
            {"symbol": "GBPUSD", "tickers": ["GBPUSD=X"]},
            {"symbol": "USDJPY", "tickers": ["USDJPY=X"]},
            {"symbol": "AUDUSD", "tickers": ["AUDUSD=X"]},
            {"symbol": "USDCAD", "tickers": ["USDCAD=X"]},
            {"symbol": "USDCHF", "tickers": ["USDCHF=X"]},
            {"symbol": "NZDUSD", "tickers": ["NZDUSD=X"]},
            {"symbol": "EURGBP", "tickers": ["EURGBP=X"]},
            {"symbol": "EURJPY", "tickers": ["EURJPY=X"]},
            {"symbol": "GBPJPY", "tickers": ["GBPJPY=X"]},
            {"symbol": "EURAUD", "tickers": ["EURAUD=X"]},
            {"symbol": "GBPAUD", "tickers": ["GBPAUD=X"]},

           # Commodities (more reliable tickers)
# Note: yfinance spot XAUUSD=X / XAGUSD=X is flaky; futures are reliable.
{"symbol": "XAUUSD", "ticker": "GC=F"},   # Gold futures
{"symbol": "XAGUSD", "ticker": "SI=F"},   # Silver futures
{"symbol": "USOIL",  "ticker": "CL=F"},
{"symbol": "NATGAS", "ticker": "NG=F"},


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

    def category(self, symbol: str) -> str:
        s = symbol.upper()
        if len(s) == 6 and s.isalpha():
            return "CURRENCIES"
        if s in ("XAUUSD", "XAGUSD", "USOIL", "NATGAS"):
            return "COMMODITIES"
        if s in ("SPX", "DJI", "NASDAQ", "RUSSELL", "FTSE", "DAX", "CAC", "NIKKEI", "HSI"):
            return "INDICES"
        if s == "DXY":
            return "DOLLAR"
        if s in ("BTCUSD", "ETHUSD"):
            return "CRYPTO"
        return "STOCKS"

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
        series = [safe_float(x) for x in close.tail(120).tolist()]

        c = close.tail(min(len(close), 180)).reset_index(drop=True)
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

        key_level = (swing_high + swing_low) / 2.0 if swing_high is not None and swing_low is not None else None

        return series, {
            "swing_high": float(swing_high) if swing_high is not None else None,
            "swing_low": float(swing_low) if swing_low is not None else None,
            "key_level": float(key_level) if key_level is not None else None,
        }

    def build_setup_projection(
        self,
        symbol: str,
        status: str,
        risk_tier: str,
        alignment: int,
        price: float,
        levels: Dict[str, Optional[float]],
    ) -> str:
        if status == "NEUTRAL" or alignment < 2:
            return "Projection: no clear multi-timeframe bias yet."

        direction = "BUY" if status == "BULLISH" else "SELL"
        sh = levels.get("swing_high")
        slw = levels.get("swing_low")
        key = levels.get("key_level")

        key_txt = round_price(symbol, key) if key is not None else "--"
        sh_txt = round_price(symbol, sh) if sh is not None else "--"
        sl_txt = round_price(symbol, slw) if slw is not None else "--"

        if direction == "BUY":
            return (
                f"{risk_tier} Projection (BUY): Bias bullish. "
                f"Watch pullback into Key {key_txt} then continuation toward Swing High {sh_txt}. "
                f"Invalidation below Swing Low {sl_txt}."
            )
        else:
            return (
                f"{risk_tier} Projection (SELL): Bias bearish. "
                f"Watch retrace into Key {key_txt} then continuation toward Swing Low {sl_txt}. "
                f"Invalidation above Swing High {sh_txt}."
            )

    def analyze_one_ticker(self, display_symbol: str, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Attempt analysis using ONE ticker. Returns None if yfinance returns empty.
        """
        raw = yf.download(
            ticker,
            period="1y",
            interval="1h",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
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

        m15_raw = yf.download(
            ticker,
            period="5d",
            interval="15m",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if m15_raw is None or m15_raw.empty or "Close" not in m15_raw.columns:
            price = 0.0
        else:
            price = safe_float(m15_raw["Close"].dropna().iloc[-1], default=0.0)

        price = float(round_price(display_symbol, float(price)))
        h4_series, h4_levels = self.make_h4_series_and_levels(raw)

        risk_tier = "A+" if alignment == 4 else "A" if alignment == 3 else "B"

        # signal BUY/SELL/WAITING (not BULLISH/BEARISH)
        if status == "NEUTRAL" or alignment < 3:
            signal = "WAITING"
        else:
            signal = "BUY" if status == "BULLISH" else "SELL"

        cat = self.category(display_symbol)
        spr = estimate_spread(display_symbol, float(price))

        setup = self.build_setup_projection(
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

    def analyze(self, display_symbol: str, tickers: List[str]) -> Optional[Dict[str, Any]]:
        """
        ✅ Try tickers in order until one works.
        """
        for t in tickers:
            try:
                res = self.analyze_one_ticker(display_symbol, t)
                if res:
                    res["source_ticker"] = t  # debug
                    return res
            except Exception:
                continue
        return None


# ---------------------------
# App state
# ---------------------------

engine = QuantEngine()

MARKET_STATE: Dict[str, Dict[str, Any]] = {}
PROGRESS: Dict[str, Any] = {"current": 0, "total": len(engine.assets), "updated_at": None}

SESSION_BRIEF: Dict[str, Any] = {
    "updated_at": None,
    "next_session": None,
    "pre_open_window": False,
    "headline": "Session Brief",
    "summary": "Initializing…",
    "confirmed_aplus": [],
    "forming_bplus": [],
    "sources": [
        {"title": "Reuters Markets", "url": "https://www.reuters.com/markets/"},
        {"title": "Bloomberg Markets", "url": "https://www.bloomberg.com/markets"},
        {"title": "Financial Times Markets", "url": "https://www.ft.com/markets"},
        {"title": "ForexFactory Calendar", "url": "https://www.forexfactory.com/calendar"},
    ],
}

def build_server_brief() -> Dict[str, Any]:
    now = utc_now()
    nxt = next_session_info(now)
    pre_open = nxt["opens_in_min"] <= 30

    items = list(MARKET_STATE.values())

    bplus = [x for x in items if x.get("alignment_val", 0) >= 2 and x.get("status") != "NEUTRAL"]
    bplus.sort(key=lambda x: x.get("alignment_val", 0), reverse=True)

    aplus = [x for x in bplus if x.get("risk_tier") == "A+" and x.get("alignment_val", 0) >= 4 and x.get("signal") != "WAITING"]

    confirmed = [
        {
            "symbol": x["symbol"],
            "status": x["status"],
            "risk_tier": x["risk_tier"],
            "alignment_val": x["alignment_val"],
            "setup": x.get("setup", ""),
            "updated_at": x.get("updated_at"),
        }
        for x in aplus[:10]
    ]

    forming = [
        {
            "symbol": x["symbol"],
            "status": x["status"],
            "risk_tier": x["risk_tier"],
            "alignment_val": x["alignment_val"],
            "setup": x.get("setup", ""),
            "updated_at": x.get("updated_at"),
        }
        for x in bplus[:20]
        if x.get("risk_tier") in ("B", "A")
    ][:12]

    bull = sum(1 for x in items if x.get("status") == "BULLISH")
    bear = sum(1 for x in items if x.get("status") == "BEARISH")

    headline = f"{SESSIONS[nxt['key']]['name']} Session Brief"
    summary = (
        f"Tone: {bull} bullish / {bear} bearish. "
        f"Confirmed A+ setups: {len(aplus)}. "
        f"Forming projections (B/A): {len([x for x in bplus if x.get('risk_tier') in ('B','A')])}. "
        f"{'Pre-open window: focus on setups forming now.' if pre_open else 'Outside pre-open window.'}"
    )

    return {
        "updated_at": utc_now_iso(),
        "next_session": nxt,
        "pre_open_window": pre_open,
        "headline": headline,
        "summary": summary,
        "confirmed_aplus": confirmed,
        "forming_bplus": forming,
        "sources": SESSION_BRIEF["sources"],
    }

async def scanner_loop():
    while True:
        PROGRESS["current"] = 0
        PROGRESS["total"] = len(engine.assets)
        PROGRESS["updated_at"] = utc_now_iso()

        for a in engine.assets:
            display = a["symbol"]
            tickers = a.get("tickers") or []
            if isinstance(tickers, str):
                tickers = [tickers]

            res = await asyncio.to_thread(engine.analyze, display, tickers)
            if res:
                MARKET_STATE[display] = res

            PROGRESS["current"] += 1
            PROGRESS["updated_at"] = utc_now_iso()

            await asyncio.sleep(0.8)

        await asyncio.sleep(60)

async def session_brief_loop():
    while True:
        try:
            SESSION_BRIEF.update(build_server_brief())
            pre_open = bool(SESSION_BRIEF.get("pre_open_window"))
            sleep_s = 60 if pre_open else 300
            await asyncio.sleep(sleep_s)
        except Exception:
            await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    t1 = asyncio.create_task(scanner_loop())
    t2 = asyncio.create_task(session_brief_loop())
    yield
    t1.cancel()
    t2.cancel()

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
        "progress": {
            "current": PROGRESS.get("current", 0),
            "total": PROGRESS.get("total", 0),
            "updated_at": PROGRESS.get("updated_at"),
        },
    }

@app.get("/api/sessions")
async def get_sessions():
    now = utc_now()
    return {
        "now_utc": now.isoformat(),
        "sessions": SESSIONS,
        "next": next_session_info(now),
    }

@app.get("/api/session-brief")
async def get_session_brief():
    if not SESSION_BRIEF.get("updated_at"):
        SESSION_BRIEF.update(build_server_brief())
    return SESSION_BRIEF

@app.get("/")
async def health():
    return {"status": "online", "active": len(MARKET_STATE), "updated_at": utc_now_iso()}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)











