#!/usr/bin/env python3
"""
BTC Options vs Polymarket — Live Probability Distribution Dashboard
====================================================================
Compares REAL-WORLD probability estimates from two independent sources:

  1. Polymarket prediction-market crowds (normalized for overround)
  2. Deribit options market (VRP-adjusted from risk-neutral to real-world)

Both are converted to the same probability measure so the "Edge" column
is a genuine apples-to-apples disagreement — not a structural artefact
of comparing risk-neutral vs physical probabilities.

Key adjustments vs naive N(d2):
  • Risk-free rate r included in d2 (standard Black-Scholes)
  • Implied vol deflated by Variance Risk Premium ratio (realized/implied)
    so the resulting probabilities reflect real-world expectations, not
    risk-neutral pricing distortions
  • Polymarket YES prices divided by their sum to remove the ~2-5% overround
  • Cubic-spline IV interpolation with linear tail extrapolation

Run:
    python3 btc_dashboard.py
"""

import os, re, json, math, threading, time, webbrowser
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests as _req

PORT        = int(os.environ.get("PORT", 8765))
REFRESH_SEC = 45
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_HV  = "https://www.deribit.com/api/v2/public/get_historical_volatility"
POLY_URL    = "https://gamma-api.polymarket.com/events"
POLY_MKTS   = "https://gamma-api.polymarket.com/markets"
MIN_OI      = 3
RISK_FREE   = 0.043        # Fed Funds effective rate (~4.3%), update periodically
DEFAULT_VRP = 0.85          # fallback: realized vol ≈ 85% of implied vol

# ── Math ──────────────────────────────────────────────────────────────────────
def ncdf(x):
    """Standard normal CDF via math.erf — IEEE 754 precise."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_d2(S, K, T, sigma, r=0.0):
    """Black-Scholes d₂ = [ln(S/K) + (r - ½σ²)T] / (σ√T)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    return (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_d1(S, K, T, sigma, r=0.0):
    """Black-Scholes d₁ = d₂ + σ√T."""
    return bs_d2(S, K, T, sigma, r) + sigma * math.sqrt(T)

def prob_above(S, K, T, sigma, r=0.0):
    """P(BTC > K at T) = N(d₂).  Use risk-neutral σ for Q-prob, VRP-adjusted σ for P-prob."""
    return ncdf(bs_d2(S, K, T, sigma, r))

# ── Cubic Spline Interpolation (pure Python) ─────────────────────────────────
def _build_spline(xs, ys):
    """Natural cubic spline.  Returns (xs, coeffs) where coeffs = [(a,b,c,d), ...]."""
    n = len(xs) - 1
    if n < 1: return xs, []
    h = [xs[i+1] - xs[i] for i in range(n)]
    alpha = [0.0] * (n + 1)
    for i in range(1, n):
        alpha[i] = 3*(ys[i+1]-ys[i])/h[i] - 3*(ys[i]-ys[i-1])/h[i-1]
    l  = [1.0] + [0.0]*n
    mu = [0.0] * (n+1)
    z  = [0.0] * (n+1)
    for i in range(1, n):
        l[i]  = 2*(xs[i+1]-xs[i-1]) - h[i-1]*mu[i-1]
        if l[i] == 0: l[i] = 1e-12
        mu[i] = h[i] / l[i]
        z[i]  = (alpha[i] - h[i-1]*z[i-1]) / l[i]
    l[n] = 1.0
    c  = [0.0] * (n+1)
    b  = [0.0] * n
    dd = [0.0] * n
    for j in range(n-1, -1, -1):
        c[j]  = z[j] - mu[j]*c[j+1]
        b[j]  = (ys[j+1]-ys[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        dd[j] = (c[j+1]-c[j]) / (3*h[j]) if h[j] else 0
    coeffs = [(ys[i], b[i], c[i], dd[i]) for i in range(n)]
    return xs, coeffs

def _eval_spline(x, xs, coeffs):
    """Evaluate cubic spline.  Linear extrapolation beyond knot range."""
    n = len(xs) - 1
    if not coeffs: return 0.6
    if x <= xs[0]:
        a, b, c, d = coeffs[0]
        return a + b * (x - xs[0])          # linear extrapolation left
    if x >= xs[-1]:
        a, b, c, d = coeffs[-1]
        dx_end = xs[-1] - xs[-2] if n > 0 else 0
        val = a + b*dx_end + c*dx_end**2 + d*dx_end**3
        slope = b + 2*c*dx_end + 3*d*dx_end**2
        return val + slope * (x - xs[-1])    # linear extrapolation right
    # Binary search for interval
    lo_i, hi_i = 0, n - 1
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        if xs[mid+1] < x: lo_i = mid + 1
        else: hi_i = mid
    a, b, c, d = coeffs[lo_i]
    dx = x - xs[lo_i]
    return a + b*dx + c*dx**2 + d*dx**3

def build_smile(calls_sorted):
    """Pre-compute cubic spline from a sorted call list.  Returns (xs, coeffs)."""
    if len(calls_sorted) < 2:
        return None
    xs = [c["strike"] for c in calls_sorted]
    ys = [c["iv"] / 100 for c in calls_sorted]
    return _build_spline(xs, ys)

def interp_iv(K, calls_sorted, smile=None):
    """Interpolate IV at strike K using cubic spline (or fallback to linear).
    Returns IV as a decimal (e.g. 0.55)."""
    if not calls_sorted: return 0.6

    # Use cubic spline if available
    if smile is not None:
        xs, coeffs = smile
        if coeffs:
            return max(_eval_spline(K, xs, coeffs), 0.05)  # floor at 5%

    # Fallback: linear interpolation
    if K <= calls_sorted[0]["strike"]:  return calls_sorted[0]["iv"] / 100
    if K >= calls_sorted[-1]["strike"]: return calls_sorted[-1]["iv"] / 100
    for i in range(len(calls_sorted) - 1):
        lo, hi = calls_sorted[i], calls_sorted[i+1]
        if lo["strike"] <= K <= hi["strike"]:
            t = (K - lo["strike"]) / (hi["strike"] - lo["strike"])
            return ((1-t)*lo["iv"] + t*hi["iv"]) / 100
    return calls_sorted[0]["iv"] / 100

# ── Deribit ───────────────────────────────────────────────────────────────────
def fetch_deribit():
    resp = _req.get(DERIBIT_URL, params={"currency": "BTC", "kind": "option"}, timeout=15)
    resp.raise_for_status()
    now = datetime.now(timezone.utc)
    by_expiry = {}
    spot = 0

    for item in resp.json().get("result", []):
        name  = item.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) != 4 or parts[3] != "C": continue
        try:
            strike    = float(parts[2])
            expiry_dt = datetime.strptime(parts[1], "%d%b%y").replace(
                tzinfo=timezone.utc, hour=8)
        except ValueError:
            continue
        days = (expiry_dt - now).days
        if days < 0 or days > 120: continue
        oi = float(item.get("open_interest") or 0)
        if oi < MIN_OI: continue
        iv = float(item.get("mark_iv") or 0)
        S  = float(item.get("underlying_price") or 0)
        if iv <= 0 or S <= 0: continue
        if S > spot: spot = S

        key = expiry_dt.strftime("%Y-%m-%d")
        if key not in by_expiry:
            by_expiry[key] = {"expiry": key, "expiry_dt": expiry_dt, "calls": []}
        by_expiry[key]["calls"].append({
            "instrument": name, "strike": strike, "iv": iv,
            "oi": oi, "underlying": S,
        })

    for k in by_expiry:
        by_expiry[k]["calls"].sort(key=lambda c: c["strike"])

    return spot, by_expiry

def fetch_realized_vol():
    """Fetch BTC annualized realized (historical) volatility from Deribit.
    Returns decimal (e.g. 0.48 = 48% annualized) or None on failure."""
    try:
        r = _req.get(DERIBIT_HV, params={"currency": "BTC"}, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", [])
        if data and len(data) > 0:
            # result is [[timestamp_ms, hv_pct], ...] sorted chronologically
            latest_hv = data[-1][1]  # annualized HV in percent
            return latest_hv / 100
    except Exception:
        pass
    return None

def compute_atm_iv(spot, by_expiry):
    """Find ATM implied vol: nearest strike to spot across the nearest expiry."""
    if not by_expiry: return None
    # Use the nearest expiry with enough data
    now = datetime.now(timezone.utc)
    nearest_key = None
    nearest_days = float("inf")
    for k, v in by_expiry.items():
        days = (v["expiry_dt"] - now).total_seconds() / 86400
        if 0 < days < nearest_days and len(v["calls"]) >= 3:
            nearest_days = days
            nearest_key = k
    if not nearest_key: return None
    calls = by_expiry[nearest_key]["calls"]
    best_dist = float("inf")
    atm_iv = None
    for c in calls:
        dist = abs(c["strike"] - spot)
        if dist < best_dist:
            best_dist = dist
            atm_iv = c["iv"] / 100
    return atm_iv

# ── Polymarket ────────────────────────────────────────────────────────────────
MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]

def candidate_slugs():
    """Generate slug candidates for the next 90 days, starting from yesterday
    so we don't miss today if the event technically hasn't closed yet."""
    now = datetime.now(timezone.utc)
    slugs = []
    for delta_days in range(-1, 90):
        d = now + timedelta(days=delta_days)
        m = MONTHS[d.month - 1]
        slugs.append(f"bitcoin-price-on-{m}-{d.day}")
    return slugs

def fetch_poly_events():
    """
    Fetch active BTC price bracket events.
    Primary: search by specific date slugs.
    Fallback: search the markets API by keyword so we show something even
    when there are no current slug-based events.
    """
    events = []
    seen   = set()

    # ── Primary: slug search ──────────────────────────────────────────────────
    for slug in candidate_slugs():
        try:
            r     = _req.get(POLY_URL, params={"slug": slug}, timeout=8)
            batch = r.json() if isinstance(r.json(), list) else []
            for ev in batch:
                eid = ev.get("id","")
                if ev.get("closed") or eid in seen: continue
                markets = ev.get("markets", [])
                if not any(
                    "price" in (m.get("question","")).lower() and
                    any(w in m.get("question","").lower()
                        for w in ("between","above","below","under"))
                    for m in markets
                ):
                    continue
                seen.add(eid)
                events.append(ev)
        except Exception:
            continue

    # ── Fallback: keyword search for open BTC price bracket markets ───────────
    if not events:
        try:
            r = _req.get(POLY_MKTS, params={
                "closed": "false",
                "tag":    "crypto",
                "limit":  100,
            }, timeout=10)
            mkts = r.json() if isinstance(r.json(), list) else []
            ev_map = {}
            for m in mkts:
                q = m.get("question","")
                if not ("bitcoin" in q.lower() or "btc" in q.lower()): continue
                if not any(w in q.lower() for w in ("between","above","below","price")):
                    continue
                gid = m.get("groupItemTitle") or m.get("slug","fallback")
                if gid not in ev_map:
                    ev_map[gid] = {
                        "id":      gid,
                        "title":   gid,
                        "slug":    gid,
                        "closed":  False,
                        "endDate": m.get("endDate",""),
                        "markets": [],
                    }
                ev_map[gid]["markets"].append(m)
            for ev in ev_map.values():
                if ev["markets"]:
                    events.append(ev)
        except Exception:
            pass

    return events

def parse_all_brackets(question):
    """Extract (lo, hi) price bounds from a bracket question string."""
    amounts = []
    for m in re.finditer(r'\$([\d,]+)', question):
        v = float(m.group(1).replace(",", ""))
        if v > 1000:
            amounts.append(v)
    q = question.lower()
    if "between" in q and len(amounts) >= 2:
        return sorted(amounts[:2])[0], sorted(amounts[:2])[1]
    elif "above" in q and len(amounts) >= 1:
        return max(amounts), None
    elif ("below" in q or "under" in q) and len(amounts) >= 1:
        return None, min(amounts)
    return None, None

# ── Core computation ──────────────────────────────────────────────────────────
def compute_data():
    t0 = time.time()
    spot, deribit = fetch_deribit()
    poly_events   = fetch_poly_events()
    now           = datetime.now(timezone.utc)

    # ── Compute Variance Risk Premium (VRP) adjustment ────────────────────────
    # ATM implied vol (what the options market charges)
    atm_iv = compute_atm_iv(spot, deribit)

    # Realized vol (what BTC actually does)
    realized_vol = fetch_realized_vol()

    # VRP ratio = realized / implied.  < 1 means IV overstates actual vol.
    # We use this to deflate IV → σ_real ≈ IV × VRP_ratio, giving real-world probs.
    if realized_vol and atm_iv and atm_iv > 0:
        vrp_ratio = realized_vol / atm_iv
    else:
        vrp_ratio = DEFAULT_VRP

    # Clamp to sane range (0.50 – 1.05); ratio > 1.0 means realized > implied
    # which can happen during vol spikes — still valid, just means IV is cheap
    vrp_ratio = max(0.50, min(1.05, vrp_ratio))

    r = RISK_FREE  # risk-free rate for d₂

    result_events = []

    for ev in poly_events:
        end_raw = ev.get("endDate","")
        try:
            end_dt = datetime.fromisoformat(end_raw.replace("Z","+00:00"))
        except Exception:
            continue

        T_years = max((end_dt - now).total_seconds() / (365.25 * 86400), 1e-9)

        best_expiry_key = None
        best_diff = float("inf")
        for k, v in deribit.items():
            diff = abs((v["expiry_dt"] - end_dt).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_expiry_key = k

        calls = deribit.get(best_expiry_key, {}).get("calls", []) if best_expiry_key else []

        # Pre-build cubic spline for this expiry's smile
        smile = build_smile(calls) if calls else None

        # ── Collect raw bracket data ──────────────────────────────────────────
        raw_brackets = []
        for m in ev.get("markets", []):
            q = m.get("question","")
            lo, hi = parse_all_brackets(q)
            if lo is None and hi is None:
                continue

            raw_p = m.get("outcomePrices","")
            poly_raw = 0.5
            try:
                prices = json.loads(raw_p) if isinstance(raw_p, str) else raw_p
                poly_raw = float(prices[0])
            except Exception:
                pass

            # Options probability: VRP-adjusted (real-world)
            if calls:
                if lo is not None and hi is not None:
                    iv_lo = interp_iv(lo, calls, smile)
                    iv_hi = interp_iv(hi, calls, smile)
                    # Deflate IV by VRP ratio → real-world sigma
                    sig_lo = iv_lo * vrp_ratio
                    sig_hi = iv_hi * vrp_ratio
                    p_lo = prob_above(spot, lo, T_years, sig_lo, r)
                    p_hi = prob_above(spot, hi, T_years, sig_hi, r)
                    opt_prob_rw = max(p_lo - p_hi, 0.0)
                    # Also compute risk-neutral for reference
                    rn_lo = prob_above(spot, lo, T_years, iv_lo, r)
                    rn_hi = prob_above(spot, hi, T_years, iv_hi, r)
                    opt_prob_rn = max(rn_lo - rn_hi, 0.0)
                elif lo is not None and hi is None:
                    iv_lo = interp_iv(lo, calls, smile)
                    opt_prob_rw = prob_above(spot, lo, T_years, iv_lo * vrp_ratio, r)
                    opt_prob_rn = prob_above(spot, lo, T_years, iv_lo, r)
                else:
                    iv_hi = interp_iv(hi, calls, smile)
                    opt_prob_rw = 1.0 - prob_above(spot, hi, T_years, iv_hi * vrp_ratio, r)
                    opt_prob_rn = 1.0 - prob_above(spot, hi, T_years, iv_hi, r)
            else:
                opt_prob_rw = None
                opt_prob_rn = None

            if lo is not None and hi is not None:
                label = f"${lo/1000:.0f}k\u2013${hi/1000:.0f}k"
                mid   = (lo + hi) / 2
            elif lo is not None:
                label = f">${lo/1000:.0f}k"
                mid   = lo * 1.05
            else:
                label = f"<${hi/1000:.0f}k"
                mid   = hi * 0.95

            raw_brackets.append({
                "label":     label,
                "lo":        lo,
                "hi":        hi,
                "mid":       mid,
                "poly_raw":  poly_raw,             # 0–1 raw YES price
                "opt_rw":    opt_prob_rw,           # 0–1 real-world adjusted
                "opt_rn":    opt_prob_rn,           # 0–1 risk-neutral
                "question":  q[:80],
                "volume":    float(m.get("volume") or 0),
            })

        raw_brackets.sort(key=lambda b: b["mid"])

        # ── Normalize Polymarket overround ────────────────────────────────────
        # Sum of raw YES prices is typically 102–105%.  Dividing by the sum
        # converts to proper probabilities that sum to 100%.
        poly_sum = sum(b["poly_raw"] for b in raw_brackets)
        overround = poly_sum  # save for display (e.g. 1.04 = 4% overround)

        brackets = []
        for b in raw_brackets:
            if poly_sum > 0:
                poly_norm = b["poly_raw"] / poly_sum   # normalized to sum=1
            else:
                poly_norm = b["poly_raw"]

            opt_rw = b["opt_rw"]
            opt_rn = b["opt_rn"]

            # Edge: real-world options prob vs normalized Polymarket prob
            edge = (opt_rw - poly_norm) if opt_rw is not None else None

            brackets.append({
                "label":       b["label"],
                "lo":          b["lo"],
                "hi":          b["hi"],
                "mid":         b["mid"],
                "poly_raw":    round(b["poly_raw"] * 100, 2),
                "poly_prob":   round(poly_norm * 100, 2),      # normalized
                "opt_prob":    round(opt_rw * 100, 2) if opt_rw is not None else None,
                "opt_prob_rn": round(opt_rn * 100, 2) if opt_rn is not None else None,
                "edge":        round(edge * 100, 2) if edge is not None else None,
                "question":    b["question"],
                "volume":      b["volume"],
            })

        if brackets:
            result_events.append({
                "title":          ev.get("title",""),
                "slug":           ev.get("slug",""),
                "end_date":       end_raw,
                "end_label":      end_dt.strftime("%b %-d, %H:%M UTC"),
                "deribit_expiry": best_expiry_key or "none",
                "brackets":       brackets,
                "overround":      round(overround * 100, 1),
                "total_poly":     round(sum(b["poly_prob"] for b in brackets), 1),
                "total_opt":      round(sum(b["opt_prob"] for b in brackets if b["opt_prob"]), 1),
                "total_opt_rn":   round(sum(b["opt_prob_rn"] for b in brackets if b["opt_prob_rn"]), 1),
            })

    # ── Vol surface (unchanged: still shows risk-neutral for vol surface) ─────
    vol_surface = []
    for key in sorted(deribit.keys()):
        for c in deribit[key]["calls"]:
            T = max((deribit[key]["expiry_dt"] - now).total_seconds() / (365.25*86400), 1e-9)
            sigma = c["iv"] / 100
            delta_c = ncdf(bs_d1(spot, c["strike"], T, sigma, r))
            p_itm   = ncdf(bs_d2(spot, c["strike"], T, sigma, r))
            vol_surface.append({
                "instrument": c["instrument"],
                "expiry":     key,
                "strike":     c["strike"],
                "iv":         c["iv"],
                "delta":      round(delta_c, 3),
                "prob_itm":   round(p_itm, 4),
                "oi":         c["oi"],
            })

    return {
        "spot":         spot,
        "updated":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "elapsed":      round(time.time() - t0, 1),
        "events":       result_events,
        "vol_surface":  vol_surface,
        "n_deribit":    len(vol_surface),
        "n_events":     len(result_events),
        "atm_iv":       round(atm_iv * 100, 1) if atm_iv else None,
        "realized_vol": round(realized_vol * 100, 1) if realized_vol else None,
        "vrp_ratio":    round(vrp_ratio, 3),
        "risk_free":    round(r * 100, 2),
    }

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache = {"data": None, "error": None, "loading": True}
_lock  = threading.Lock()

def _loop():
    while True:
        try:
            d = compute_data()
            with _lock: _cache.update({"data": d, "error": None, "loading": False})
        except Exception as e:
            with _lock: _cache.update({"error": str(e), "loading": False})
        time.sleep(REFRESH_SEC)

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>BTC Options vs Polymarket</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#eef2f7;
  color:#1e293b;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  font-size:15px;
  min-height:100vh;
  line-height:1.5;
}

/* ── Info icon ────────────────────────────────────────────────────────────── */
.info-tip{
  display:inline-flex;align-items:center;justify-content:center;
  width:18px;height:18px;border-radius:50%;
  background:#dbeafe;color:#2563eb;
  font-size:11px;font-style:italic;font-weight:800;
  cursor:help;vertical-align:middle;margin-left:5px;
  flex-shrink:0;user-select:none;
  border:1.5px solid #bfdbfe;
}

/* ── JS tooltip element ───────────────────────────────────────────────────── */
#jtip{
  position:fixed;z-index:99999;
  background:#1e293b;color:#f1f5f9;
  padding:12px 15px;border-radius:9px;
  font-size:13px;font-weight:400;font-style:normal;
  max-width:310px;width:max-content;
  line-height:1.6;text-align:left;
  pointer-events:none;
  opacity:0;transition:opacity .15s;
  box-shadow:0 6px 24px rgba(0,0,0,.22);
  word-break:normal;
}

/* ── Top bar ──────────────────────────────────────────────────────────────── */
.topbar{
  background:#ffffff;
  border-bottom:2px solid #e2e8f0;
  padding:14px 28px;
  display:flex;align-items:center;gap:22px;flex-wrap:wrap;
  position:sticky;top:0;z-index:100;
  box-shadow:0 2px 10px rgba(0,0,0,.07);
}
.topbar h1{font-size:20px;font-weight:800;color:#0f172a;letter-spacing:-.01em;white-space:nowrap}
.stat{display:flex;flex-direction:column;gap:2px}
.stat label{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.09em;font-weight:600;display:flex;align-items:center}
.stat span{font-size:17px;font-weight:700;color:#0f172a}
#countdown{margin-left:auto;font-size:14px;color:#94a3b8;white-space:nowrap;display:flex;align-items:center;gap:7px}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;display:inline-block;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}

/* ── Layout ───────────────────────────────────────────────────────────────── */
.main{padding:22px 28px;display:grid;gap:20px}

.card{
  background:#ffffff;
  border:1.5px solid #e2e8f0;
  border-radius:12px;
  box-shadow:0 2px 8px rgba(0,0,0,.05);
}
.card-hdr{
  padding:14px 18px;
  border-bottom:1.5px solid #f1f5f9;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;
  background:#f8fafc;
  border-radius:12px 12px 0 0;
}
.card-hdr h2{font-size:14px;text-transform:uppercase;letter-spacing:.07em;color:#475569;font-weight:800}

/* ── Event tabs ───────────────────────────────────────────────────────────── */
.event-tabs{
  display:flex;gap:9px;flex-wrap:wrap;
  padding:14px 18px;
  border-bottom:1.5px solid #f1f5f9;
  background:#fafbfd;
}
.etab{
  padding:7px 18px;border-radius:22px;font-size:14px;font-weight:600;
  cursor:pointer;border:1.5px solid #e2e8f0;color:#64748b;
  background:#fff;transition:all .15s;
}
.etab:hover{border-color:#93c5fd;color:#2563eb;background:#eff6ff}
.etab.active{background:#2563eb;border-color:#2563eb;color:#ffffff;box-shadow:0 2px 8px rgba(37,99,235,.3)}

/* ── Grid ─────────────────────────────────────────────────────────────────── */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.chart-wrap{padding:18px;height:400px;position:relative}
.chart-wrap-sm{padding:18px;height:310px;position:relative}

/* ── Table ────────────────────────────────────────────────────────────────── */
table{width:100%;border-collapse:collapse}
th{
  padding:12px 16px;
  font-size:13px;text-transform:uppercase;letter-spacing:.06em;
  color:#475569;border-bottom:2px solid #e2e8f0;
  text-align:right;background:#f8fafc;font-weight:700;
  white-space:nowrap;
}
th:first-child{text-align:left}
td{padding:12px 16px;border-bottom:1px solid #f1f5f9;font-size:14px;text-align:right;font-weight:500}
td:first-child{text-align:left}
tr:last-child td{border:none}
tr:hover td{background:#f8fafc}

/* ── Signal pills ─────────────────────────────────────────────────────────── */
.under{color:#16a34a;font-weight:700}
.over{color:#dc2626;font-weight:700}
.fair{color:#94a3b8;font-weight:600}
.pill{display:inline-block;padding:4px 12px;border-radius:5px;font-size:13px;font-weight:700;letter-spacing:.03em}
.pill-under{background:#dcfce7;color:#16a34a}
.pill-over{background:#fee2e2;color:#dc2626}
.pill-fair{background:#f1f5f9;color:#94a3b8}

/* ── Legend bar ───────────────────────────────────────────────────────────── */
.legend-bar{
  display:flex;align-items:center;gap:22px;flex-wrap:wrap;
  padding:12px 18px;
  border-top:1.5px solid #f1f5f9;
  background:#fafbfd;
  font-size:14px;font-weight:600;
}
.legend-item{display:flex;align-items:center;gap:8px}
.legend-swatch{width:24px;height:15px;border-radius:4px;flex-shrink:0}
.legend-line-swatch{width:28px;height:3px;border-radius:2px;flex-shrink:0}

/* ── Totals bar ───────────────────────────────────────────────────────────── */
.totals{
  display:flex;gap:24px;padding:12px 18px;
  background:#f8fafc;font-size:14px;color:#64748b;
  border-top:1.5px solid #f1f5f9;flex-wrap:wrap;align-items:center;
}
.totals b{color:#0f172a;font-size:16px}

/* ── Loading ──────────────────────────────────────────────────────────────── */
#loading{
  position:fixed;inset:0;background:#eef2f7;
  display:flex;align-items:center;justify-content:center;
  flex-direction:column;gap:14px;z-index:200;
}
.spin{width:40px;height:40px;border:3px solid #e2e8f0;border-top-color:#2563eb;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{padding:36px;text-align:center;color:#94a3b8;font-size:15px}

/* ── Insight box ──────────────────────────────────────────────────────────── */
.insight-box{
  background:#fefce8;border:1.5px solid #fde68a;border-radius:9px;
  padding:13px 17px;margin:14px 18px 0 18px;
  font-size:13px;color:#78350f;line-height:1.65;
}
.insight-box strong{color:#92400e;font-size:14px}

/* ── Methodology box ──────────────────────────────────────────────────────── */
.method-box{
  background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:9px;
  padding:13px 17px;margin:14px 18px 0 18px;
  font-size:13px;color:#14532d;line-height:1.65;
}
.method-box strong{color:#166534;font-size:14px}

/* ── No-events notice ─────────────────────────────────────────────────────── */
.no-events-notice{
  background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:9px;
  padding:16px 20px;margin:16px 18px;
  font-size:14px;color:#1e40af;line-height:1.7;
}
.no-events-notice strong{color:#1d4ed8}
</style>
</head>
<body>

<div id="jtip"></div>

<div id="loading">
  <div class="spin"></div>
  <div style="color:#64748b;font-size:15px;margin-top:6px">Fetching Deribit + Polymarket&#8230;</div>
</div>

<div class="topbar">
  <h1>&#9889; BTC Options vs Polymarket</h1>
  <div class="stat">
    <label>BTC Spot <span class="info-tip" data-tip="Current Bitcoin spot price from Deribit underlying. Used as S in all Black-Scholes calculations.">i</span></label>
    <span id="spot">&#8212;</span>
  </div>
  <div class="stat">
    <label>ATM IV <span class="info-tip" data-tip="At-the-money implied volatility from the nearest Deribit expiry. This is what the options market CHARGES for volatility — it includes a risk premium above what BTC actually realizes. Example: ATM IV = 58% means options are priced as if BTC will move at 58% annualized vol.">i</span></label>
    <span id="atm-iv">&#8212;</span>
  </div>
  <div class="stat">
    <label>Realized Vol <span class="info-tip" data-tip="BTC's actual historical volatility from Deribit. This is what BTC ACTUALLY DID — no risk premium. Example: RV = 48% means BTC has been moving at 48% annualized. Compare to ATM IV to see the Variance Risk Premium.">i</span></label>
    <span id="realized-vol">&#8212;</span>
  </div>
  <div class="stat">
    <label>VRP Adj. <span class="info-tip" data-tip="Variance Risk Premium adjustment ratio = Realized Vol / ATM IV. Used to deflate implied vol down to real-world vol so that options probabilities reflect what the market EXPECTS to happen, not the inflated risk-neutral pricing. Example: VRP = 0.83 means realized vol is 83% of implied, so we multiply all IVs by 0.83 before computing probabilities.">i</span></label>
    <span id="vrp-ratio">&#8212;</span>
  </div>
  <div class="stat">
    <label>Contracts <span class="info-tip" data-tip="Number of Deribit call contracts loaded (OI >= 3, expiry <= 120d).">i</span></label>
    <span id="nd">&#8212;</span>
  </div>
  <div class="stat">
    <label>Events <span class="info-tip" data-tip="Active Polymarket BTC price-bracket events found.">i</span></label>
    <span id="ne">&#8212;</span>
  </div>
  <div class="stat">
    <label>Updated</label>
    <span id="upd" style="font-size:13px;color:#94a3b8;font-weight:400">&#8212;</span>
  </div>
  <div id="countdown"><span class="dot"></span>next refresh in <b id="ct" style="color:#0f172a;font-size:15px">&#8212;</b>s</div>
</div>

<div class="main">

  <!-- ─── Methodology note ─── -->
  <div class="method-box">
    <strong>Apples-to-Apples Methodology:</strong> Both columns show <em>real-world</em> probability estimates.
    Polymarket YES prices are <strong>normalized</strong> (divided by their sum) to remove the ~2-5% overround/house edge.
    Options probabilities are <strong>VRP-adjusted</strong> &mdash; implied vol is deflated by the Realized Vol / ATM IV ratio
    to convert from risk-neutral pricing to real-world expectations. The risk-free rate (r=<span id="rfr">4.3</span>%)
    is included in d&#8322;. The result: both numbers answer the same question &mdash;
    <em>&ldquo;what is the probability BTC lands in this bracket?&rdquo;</em> &mdash; from two independent crowds.
  </div>

  <!-- ─── Probability Comparison Card ─── -->
  <div class="card">
    <div class="card-hdr">
      <h2>Real-World Probability Comparison</h2>
      <span class="info-tip" data-tip="Both sources are adjusted to real-world probabilities so the comparison is apples-to-apples. Polymarket: normalized for overround. Options: VRP-deflated IV with risk-free rate. A gap between the two is a genuine disagreement between prediction-market crowds and derivatives traders about the same thing.">i</span>
      <span id="event-meta" style="font-size:13px;color:#94a3b8;margin-left:auto"></span>
    </div>

    <div class="event-tabs" id="event-tabs">
      <span style="color:#94a3b8;font-size:14px;padding:4px">Loading events&#8230;</span>
    </div>

    <div id="no-events-panel" style="display:none">
      <div class="no-events-notice">
        <strong>No active Polymarket BTC price events found.</strong><br>
        Today's event has likely already settled. Polymarket typically lists new
        daily events a few hours before midnight UTC. Check back soon &mdash; the
        dashboard will automatically pick them up on the next refresh.<br><br>
        The Deribit vol surface and smile charts below are still live.
      </div>
    </div>

    <div class="chart-wrap" id="main-chart-wrap">
      <canvas id="mainChart"></canvas>
    </div>

    <!-- Color-matched legend -->
    <div class="legend-bar">
      <div class="legend-item">
        <div class="legend-swatch" style="background:rgba(37,99,235,0.70);border:1.5px solid #2563eb"></div>
        <span style="color:#2563eb">Polymarket (normalized)</span>
        <span class="info-tip" data-tip="Polymarket YES price divided by the sum of all brackets' YES prices. This removes the ~2-5% overround (house edge) so probabilities sum to 100%. Example: raw YES = 18% but brackets sum to 104%, so normalized = 18/104 = 17.3%. This is the crowd's true implied probability.">i</span>
      </div>
      <div class="legend-item">
        <div class="legend-line-swatch" style="background:#7c3aed"></div>
        <span style="color:#7c3aed">Options (real-world)</span>
        <span class="info-tip" data-tip="Real-world probability from Deribit options. IV is deflated by the VRP ratio (Realized Vol / ATM IV) before computing Black-Scholes N(d2) with the risk-free rate. This converts risk-neutral pricing probabilities into real-world expectations — what the derivatives market actually THINKS will happen, stripped of the volatility risk premium.">i</span>
      </div>
      <div style="margin-left:auto;font-size:13px;color:#94a3b8;font-weight:500">
        Edge = Options (adj.) &#8722; Polymarket (norm.)
      </div>
    </div>

    <div class="totals">
      <div>Poly (norm.): <b id="poly-total">&#8212;</b>%
        <span class="info-tip" data-tip="Sum of normalized Polymarket probabilities. Should be ~100% by construction (since we normalize). Minor rounding may cause small deviations.">i</span>
      </div>
      <div>Options (adj.): <b id="opt-total">&#8212;</b>%
        <span class="info-tip" data-tip="Sum of VRP-adjusted options probabilities across all brackets. Deviations from 100% indicate the brackets don't cover all probability mass (missing tails) or minor interpolation imprecision.">i</span>
      </div>
      <div id="overround-display" style="font-size:13px;color:#94a3b8">
        Poly overround: <b id="poly-overround" style="color:#0f172a">&#8212;</b>%
        <span class="info-tip" data-tip="Raw Polymarket YES prices summed before normalization. Values above 100% represent the overround/house edge — the extra % the market maker extracts. Example: 104% overround means traders collectively overpay by 4%.">i</span>
      </div>
    </div>

    <!-- Bracket comparison table -->
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th style="text-align:left">
            Bracket
            <span class="info-tip" data-tip="The BTC price range this row covers. Example: '$82k-$84k' resolves YES if BTC is between $82,000 and $84,000 at expiry.">i</span>
          </th>
          <th>
            <span style="color:#2563eb">Poly (norm.)</span>
            <span class="info-tip" data-tip="Polymarket YES price AFTER overround normalization. Raw price divided by sum of all brackets' prices, so the column sums to ~100%. This is the crowd's real-world probability estimate, cleaned of house edge.">i</span>
          </th>
          <th>
            <span style="color:#7c3aed">Options (adj.)</span>
            <span class="info-tip" data-tip="Real-world probability from options. IV is deflated by VRP ratio (RV/IV) and risk-free rate is included in d2. This strips out the variance risk premium so the number reflects what derivatives traders actually expect, not what they charge for insurance.">i</span>
          </th>
          <th>
            Edge
            <span class="info-tip" data-tip="Options (adjusted) minus Polymarket (normalized). Both are real-world probabilities, so this edge is a genuine disagreement between two crowds. Positive (green) = options traders think this bracket is more likely than Polymarket crowds. Negative (red) = less likely. Unlike the old N(d2) edge, this is NOT inflated by the variance risk premium.">i</span>
          </th>
          <th>
            Signal
            <span class="info-tip" data-tip="Trading signal based on edge magnitude. UNDER (green) = edge > +3pp, options crowd disagrees with Polymarket — potential buy YES. OVER (red) = edge < -3pp — potential buy NO. FAIR = within +/-3pp, both crowds roughly agree.">i</span>
          </th>
          <th>
            Volume ($)
            <span class="info-tip" data-tip="USD volume traded on this Polymarket bracket. Higher = more reliable price.">i</span>
          </th>
        </tr></thead>
        <tbody id="bracket-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ─── Smile Charts ─── -->
  <div class="grid2">

    <div class="card">
      <div class="card-hdr">
        <h2>Options Probability Smile</h2>
        <span class="info-tip" data-tip="N(d2) for each strike — the risk-neutral probability that BTC finishes ABOVE that strike at expiry. The vol surface table below still uses risk-neutral values since these are the standard for derivatives analysis.">i</span>
        <span id="smile-expiry" style="font-size:13px;color:#94a3b8;margin-left:auto"></span>
      </div>
      <div class="insight-box">
        <strong>What this tells you:</strong> The probability the options market assigns to BTC finishing <em>above</em> each strike price. Where the curve crosses 50% is the market's expected settlement price. Asymmetry reveals directional bias &mdash; a steep left drop means crash risk is heavily priced.
      </div>
      <div class="chart-wrap-sm"><canvas id="smileChart"></canvas></div>
    </div>

    <div class="card">
      <div class="card-hdr">
        <h2>Implied Volatility Smile</h2>
        <span class="info-tip" data-tip="Implied volatility (IV) at each strike from Deribit. The lowest point (near ATM) is the market's baseline vol forecast. The VRP adjustment is NOT applied here — this shows raw market IV. Compare ATM IV to Realized Vol in the top bar to see the risk premium.">i</span>
        <span id="iv-expiry" style="font-size:13px;color:#94a3b8;margin-left:auto"></span>
      </div>
      <div class="insight-box">
        <strong>What this tells you:</strong> The market's expected BTC volatility at each strike. The lowest point is the baseline vol forecast. A steep left wing means crash insurance is expensive. Compare the trough to the Realized Vol stat above &mdash; the gap IS the variance risk premium that the dashboard adjusts for.
      </div>
      <div class="chart-wrap-sm"><canvas id="ivChart"></canvas></div>
    </div>

  </div>

  <!-- ─── Vol Surface Table ─── -->
  <div class="card">
    <div class="card-hdr">
      <h2>Deribit Vol Surface &#8212; Calls</h2>
      <span class="info-tip" data-tip="Full table of liquid BTC calls from Deribit. Delta and P(ITM) here use RISK-NEUTRAL values (standard for derivatives). The VRP adjustment is only applied in the probability comparison above.">i</span>
      <select id="expiry-sel" style="background:#fff;color:#374151;border:1.5px solid #e2e8f0;border-radius:7px;padding:5px 12px;font-size:13px;cursor:pointer;margin-left:auto;font-weight:600">
        <option value="all">All expirations</option>
      </select>
    </div>
    <div style="overflow-x:auto;max-height:420px;overflow-y:auto">
      <table>
        <thead><tr>
          <th style="text-align:left">
            Instrument
            <span class="info-tip" data-tip="Deribit ticker: BTC-DDMMMYY-STRIKE-C.">i</span>
          </th>
          <th>
            Strike
            <span class="info-tip" data-tip="Price at which the call holder can buy BTC at expiry.">i</span>
          </th>
          <th>
            Expiry
            <span class="info-tip" data-tip="Settlement date at 08:00 UTC.">i</span>
          </th>
          <th>
            Delta
            <span class="info-tip" data-tip="N(d1) — change in option value per $1 move in BTC. Risk-neutral. ATM delta is near 0.50.">i</span>
          </th>
          <th>
            P(ITM)
            <span class="info-tip" data-tip="Risk-neutral N(d2) probability BTC finishes above this strike. Includes risk-free rate. Green > 50%, Red < 50%.">i</span>
          </th>
          <th>
            IV
            <span class="info-tip" data-tip="Implied Volatility — annualised vol backed out from Deribit option prices. Includes the variance risk premium.">i</span>
          </th>
          <th>
            OI
            <span class="info-tip" data-tip="Open Interest in BTC. Higher = more liquid and reliable.">i</span>
          </th>
        </tr></thead>
        <tbody id="surface-body"></tbody>
      </table>
    </div>
  </div>

</div><!-- /main -->

<script>
const REFRESH = REFRESH_PLACEHOLDER;
let countdown = REFRESH;
let allData = null;
let activeEventIdx = 0;
let mainChart, smileChart, ivChart;

/* ── JS Tooltip ───────────────────────────────────────────────────────────── */
(function(){
  const tip = document.getElementById('jtip');
  function pos(e){
    const pad=10, w=Math.min(tip.offsetWidth||310,310), h=tip.offsetHeight||60;
    let x=e.clientX - w/2, y=e.clientY - h - 14;
    if(x<pad) x=pad;
    if(x+w>window.innerWidth-pad) x=window.innerWidth-w-pad;
    if(y<pad) y=e.clientY+22;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }
  document.addEventListener('mouseover',e=>{
    const el=e.target.closest&&e.target.closest('.info-tip');
    if(el&&el.dataset.tip){ tip.textContent=el.dataset.tip; tip.style.opacity='1'; pos(e); }
  });
  document.addEventListener('mousemove',e=>{ if(tip.style.opacity==='1') pos(e); });
  document.addEventListener('mouseout',e=>{
    if(e.target.closest&&e.target.closest('.info-tip')) tip.style.opacity='0';
  });
})();

/* ── Chart helpers ────────────────────────────────────────────────────────── */
const gc='#f1f5f9', tc='#64748b';
const baseOpts=(yLabel,xLabel)=>({
  responsive:true, maintainAspectRatio:false,
  plugins:{
    legend:{display:false},
    tooltip:{
      backgroundColor:'#1e293b',titleColor:'#f8fafc',bodyColor:'#cbd5e1',
      borderColor:'#334155',borderWidth:1,
      titleFont:{size:14,weight:'600'},bodyFont:{size:13},padding:12
    }
  },
  scales:{
    x:{ticks:{color:tc,font:{size:13},maxRotation:38},grid:{color:gc},
       title:{display:!!xLabel,text:xLabel||'',color:tc,font:{size:13,weight:'600'}}},
    y:{ticks:{color:tc,font:{size:13},callback:v=>v+'%'},grid:{color:gc},
       title:{display:true,text:yLabel,color:tc,font:{size:13,weight:'600'}}}
  }
});

function fmt$(n){ return '$'+Number(n).toLocaleString('en-US',{maximumFractionDigits:0}); }

/* ── Render event ─────────────────────────────────────────────────────────── */
function renderEvent(events,idx){
  if(!events||!events.length) return;
  const ev=events[idx];
  document.getElementById('event-meta').textContent=
    `expires ${ev.end_label}  \u00b7  matched Deribit expiry: ${ev.deribit_expiry}`;
  document.getElementById('poly-total').textContent=ev.total_poly;
  document.getElementById('opt-total').textContent=ev.total_opt||'\u2014';
  document.getElementById('poly-overround').textContent=ev.overround||'\u2014';

  const bs=ev.brackets;
  const labels=bs.map(b=>b.label);
  const polyData=bs.map(b=>b.poly_prob);
  const optData=bs.map(b=>b.opt_prob??null);

  const chartData={
    labels,
    datasets:[
      {label:'Polymarket (normalized)',data:polyData,type:'bar',
       backgroundColor:'rgba(37,99,235,0.70)',borderColor:'rgba(37,99,235,0.95)',
       borderWidth:1.5,borderRadius:5},
      {label:'Options (real-world)',data:optData,type:'line',
       backgroundColor:'transparent',borderColor:'rgba(124,58,237,0.95)',
       borderWidth:2.5,tension:0.35,pointRadius:5,
       pointBackgroundColor:'rgba(124,58,237,0.9)',pointHoverRadius:7,
       fill:false,yAxisID:'y'}
    ]
  };

  const chartOpts={
    ...baseOpts('Probability %','Price bracket'),
    plugins:{
      tooltip:{
        backgroundColor:'#1e293b',titleColor:'#f8fafc',bodyColor:'#cbd5e1',
        borderColor:'#334155',borderWidth:1,
        titleFont:{size:14,weight:'600'},bodyFont:{size:13},padding:12,
        callbacks:{
          afterBody:(items)=>{
            const i=items[0].dataIndex, b=bs[i];
            if(b.edge!==null){
              const sign=b.edge>=0?'+':'';
              return[`Edge: ${sign}${b.edge}%`,
                b.edge>3  ?'\u2b06 Options crowd says HIGHER prob':
                b.edge<-3 ?'\u2b07 Options crowd says LOWER prob':'\u007e Both crowds agree'];
            }
            return[];
          }
        }
      },
      legend:{
        display:true,position:'top',
        labels:{font:{size:14,weight:'600'},padding:18,usePointStyle:true,
          generateLabels:(chart)=>chart.data.datasets.map((ds,i)=>({
            text:ds.label,
            fillStyle:ds.type==='line'?'transparent':ds.backgroundColor,
            strokeStyle:ds.borderColor,
            lineWidth:ds.type==='line'?2.5:0,
            color:ds.borderColor,
            hidden:false,datasetIndex:i,
            pointStyle:ds.type==='line'?'line':'rect'
          }))
        }
      }
    }
  };

  if(mainChart){mainChart.data=chartData;mainChart.options=chartOpts;mainChart.update();}
  else mainChart=new Chart(document.getElementById('mainChart').getContext('2d'),
    {type:'bar',data:chartData,options:chartOpts});

  const rows=bs.map(b=>{
    const ec=b.edge>3?'under':b.edge<-3?'over':'fair';
    const sig=b.edge>3?'UNDER':b.edge<-3?'OVER':'FAIR';
    const sign=b.edge>=0?'+':'';
    return`<tr>
      <td style="text-align:left;font-weight:700;color:#374151;font-size:15px">${b.label}</td>
      <td style="color:#2563eb;font-weight:700">${b.poly_prob.toFixed(1)}%</td>
      <td style="color:#7c3aed;font-weight:700">${b.opt_prob!==null?b.opt_prob.toFixed(1)+'%':'\u2014'}</td>
      <td class="${ec}">${b.edge!==null?sign+b.edge.toFixed(1)+'%':'\u2014'}</td>
      <td><span class="pill pill-${ec}">${sig}</span></td>
      <td style="color:#94a3b8">$${Number(b.volume).toLocaleString('en-US',{maximumFractionDigits:0})}</td>
    </tr>`;
  }).join('');
  document.getElementById('bracket-tbody').innerHTML=rows;
}

/* ── Render smiles ────────────────────────────────────────────────────────── */
function renderSmiles(surface,targetExpiry){
  const exps=[...new Set(surface.map(o=>o.expiry))].sort();
  const exp=(targetExpiry&&exps.includes(targetExpiry))?targetExpiry
            :exps.find(e=>e>=new Date().toISOString().slice(0,10))||exps[0];

  const sel=document.getElementById('expiry-sel');
  while(sel.options.length>1) sel.remove(1);
  exps.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;sel.appendChild(o);});

  const slice=surface.filter(o=>o.expiry===exp);
  document.getElementById('smile-expiry').textContent=exp;
  document.getElementById('iv-expiry').textContent=exp;

  const labs=slice.map(o=>fmt$(o.strike));
  const probs=slice.map(o=>+(o.prob_itm*100).toFixed(1));
  const ivs=slice.map(o=>+o.iv.toFixed(1));

  const mkDs=(data,color)=>({data,borderColor:color,
    backgroundColor:color.replace('rgb(','rgba(').replace(')',', 0.07)'),
    tension:0.3,pointRadius:3,pointHoverRadius:7,fill:true,borderWidth:2.5});
  const shTip={backgroundColor:'#1e293b',titleColor:'#f8fafc',bodyColor:'#cbd5e1',
    borderColor:'#334155',borderWidth:1,
    titleFont:{size:14,weight:'600'},bodyFont:{size:13},padding:12};

  if(smileChart){smileChart.data.labels=labs;smileChart.data.datasets[0].data=probs;smileChart.update();}
  else smileChart=new Chart(document.getElementById('smileChart').getContext('2d'),
    {type:'line',data:{labels:labs,datasets:[mkDs(probs,'rgb(124,58,237)')]},
     options:{...baseOpts('P(ITM) %','Strike'),plugins:{legend:{display:false},tooltip:shTip}}});

  if(ivChart){ivChart.data.labels=labs;ivChart.data.datasets[0].data=ivs;ivChart.update();}
  else ivChart=new Chart(document.getElementById('ivChart').getContext('2d'),
    {type:'line',data:{labels:labs,datasets:[mkDs(ivs,'rgb(5,150,105)')]},
     options:{...baseOpts('Implied Vol %','Strike'),plugins:{legend:{display:false},tooltip:shTip}}});
}

/* ── Render vol surface table ─────────────────────────────────────────────── */
function renderSurface(surface,filter){
  const rows=surface
    .filter(o=>filter==='all'||o.expiry===filter)
    .map(o=>`<tr>
      <td style="text-align:left;font-size:13px;color:#4f46e5;font-weight:600">${o.instrument}</td>
      <td style="font-weight:700">${fmt$(o.strike)}</td>
      <td style="color:#64748b">${o.expiry}</td>
      <td style="color:#2563eb;font-weight:700">${o.delta.toFixed(3)}</td>
      <td style="color:${o.prob_itm>.5?'#16a34a':'#dc2626'};font-weight:700">${(o.prob_itm*100).toFixed(1)}%</td>
      <td style="color:#7c3aed;font-weight:700">${o.iv.toFixed(1)}%</td>
      <td style="color:#94a3b8">${o.oi.toFixed(0)}</td>
    </tr>`).join('');
  document.getElementById('surface-body').innerHTML=rows||'<tr><td colspan="7" class="empty">No data</td></tr>';
}

/* ── Main render ──────────────────────────────────────────────────────────── */
function render(d){
  document.getElementById('loading').style.display='none';
  document.getElementById('spot').textContent='$'+d.spot.toLocaleString('en-US',{maximumFractionDigits:0});
  document.getElementById('nd').textContent=d.n_deribit;
  document.getElementById('ne').textContent=d.n_events;
  document.getElementById('upd').textContent=d.updated+' ('+d.elapsed+'s)';

  // VRP stats
  document.getElementById('atm-iv').textContent=d.atm_iv?d.atm_iv+'%':'\u2014';
  document.getElementById('realized-vol').textContent=d.realized_vol?d.realized_vol+'%':'\u2014';
  document.getElementById('vrp-ratio').textContent=d.vrp_ratio||'\u2014';
  document.getElementById('rfr').textContent=d.risk_free||'4.3';

  const tabsEl=document.getElementById('event-tabs');
  const noEvPanel=document.getElementById('no-events-panel');
  const mainWrap=document.getElementById('main-chart-wrap');
  tabsEl.innerHTML='';

  if(!d.events.length){
    noEvPanel.style.display='block';
    mainWrap.style.display='none';
    document.getElementById('bracket-tbody').innerHTML='';
  } else {
    noEvPanel.style.display='none';
    mainWrap.style.display='block';

    d.events.forEach((ev,i)=>{
      const btn=document.createElement('button');
      btn.className='etab'+(i===activeEventIdx?' active':'');
      btn.textContent=ev.end_label;
      btn.onclick=()=>{
        activeEventIdx=i;
        document.querySelectorAll('.etab').forEach((b,j)=>
          b.className='etab'+(j===i?' active':''));
        renderEvent(d.events,i);
        renderSmiles(d.vol_surface,d.events[i].deribit_expiry);
      };
      tabsEl.appendChild(btn);
    });

    if(activeEventIdx>=d.events.length) activeEventIdx=0;
    renderEvent(d.events,activeEventIdx);
  }

  const smileExpiry=d.events.length?d.events[activeEventIdx]?.deribit_expiry:null;
  renderSmiles(d.vol_surface,smileExpiry);
  renderSurface(d.vol_surface,'all');

  document.getElementById('expiry-sel').addEventListener('change',e=>{
    renderSurface(d.vol_surface,e.target.value);
  });
}

/* ── Poll loop ────────────────────────────────────────────────────────────── */
async function load(){
  try{
    const res=await fetch('/api/data');
    const j=await res.json();
    if(j.error){document.getElementById('upd').textContent='Error: '+j.error;return;}
    allData=j.data;
    render(allData);
  }catch(e){console.error(e);}
}
function tick(){
  countdown--;
  document.getElementById('ct').textContent=countdown;
  if(countdown<=0){countdown=REFRESH;load();}
}
load();
setInterval(tick,1000);
</script>
</body>
</html>""".replace("REFRESH_PLACEHOLDER", str(REFRESH_SEC))

# ── Server ────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/":
            self._respond(200, "text/html; charset=utf-8", HTML.encode())
        elif self.path == "/api/data":
            with _lock:
                body = json.dumps({"data": _cache["data"], "error": _cache["error"]}).encode()
            self._respond(200, "application/json", body)
        else:
            self._respond(404, "text/plain", b"Not found")
    def _respond(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    print("Starting data fetch...")
    threading.Thread(target=_loop, daemon=True).start()
    host = "0.0.0.0"  # bind all interfaces for cloud deployment
    url = f"http://localhost:{PORT}"
    print(f"Dashboard -> {url}   (Ctrl-C to stop)")
    # Only auto-open browser locally (not on Render/cloud)
    if not os.environ.get("RENDER"):
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    try: HTTPServer((host, PORT), Handler).serve_forever()
    except KeyboardInterrupt: print("\nStopped.")
