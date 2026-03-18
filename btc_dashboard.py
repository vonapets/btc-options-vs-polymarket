#!/usr/bin/env python3
"""
BTC Options vs Polymarket — Live Probability Distribution Dashboard
====================================================================
Compares RISK-NEUTRAL probability estimates from two independent sources:

  1. Polymarket prediction-market crowds  — live CLOB midpoints, normalized
  2. Deribit options market               — Breeden-Litzenberger model-free

Methodology v2.0 — Arbitrage-free & statistically rigorous:
  * Breeden-Litzenberger: P(K1<S_T<K2) = e^{rT}·[∂C/∂K(K2) − ∂C/∂K(K1)]
  * Exact analytical ∂C/∂K via chain rule on cubic IV spline:
      ∂C/∂K = -e^{-rT}·N(d2) + S·√T·φ(d1)·σ'(K)
    where σ'(K) is the exact polynomial derivative of the spline segment.
    No finite-difference step, no hyperparameter ΔK.
  * Combined OTM put+call smile (puts below ATM, calls above ATM)
  * Temporal variance interpolation between Deribit expiries
  * Tail rows: explicit P(<min_bracket) and P(>max_bracket) from B-L
  * No-arbitrage PDF check: flags smile segments with negative density
  * Polymarket: live CLOB midpoints (clob.polymarket.com) with bid/ask spread,
    normalized for ~2-5% overround. Falls back to Gamma outcomePrices.
  * Coinbase spot price (Polymarket's settlement reference) with Deribit basis
  * Risk-free rate r = 4.3% (Fed Funds) in d2

Run:
    python3 btc_dashboard.py
"""

import os, re, json, math, threading, time, webbrowser
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests as _req

PORT         = int(os.environ.get("PORT", 8765))
REFRESH_SEC  = 45
DERIBIT_URL  = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_HV   = "https://www.deribit.com/api/v2/public/get_historical_volatility"
COINBASE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
CLOB_BASE    = "https://clob.polymarket.com"
POLY_URL     = "https://gamma-api.polymarket.com/events"
POLY_MKTS    = "https://gamma-api.polymarket.com/markets"
MIN_OI       = 3
RISK_FREE    = 0.043   # Fed Funds effective rate (~4.3%)

# ── Math ──────────────────────────────────────────────────────────────────────
def ncdf(x):
    """Standard normal CDF via math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def npdf(x):
    """Standard normal PDF φ(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_d1(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_d2(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    return (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_call_price(S, K, T, sigma, r=0.0):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0: return max(S - K * math.exp(-r * T), 0.0)
    d1 = bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)

def prob_above(S, K, T, sigma, r=0.0):
    """P(BTC > K at T) = N(d₂). Risk-neutral."""
    return ncdf(bs_d2(S, K, T, sigma, r))

# ── Cubic Spline (pure Python) ────────────────────────────────────────────────
def _build_spline(xs, ys):
    """Natural cubic spline. Returns (xs, coeffs) where coeffs = [(a,b,c,d), ...]."""
    n = len(xs) - 1
    if n < 1: return xs, []
    h = [xs[i+1] - xs[i] for i in range(n)]
    alpha = [0.0] * (n + 1)
    for i in range(1, n):
        alpha[i] = 3*(ys[i+1]-ys[i])/h[i] - 3*(ys[i]-ys[i-1])/h[i-1]
    l = [1.0] + [0.0]*n; mu = [0.0]*(n+1); z = [0.0]*(n+1)
    for i in range(1, n):
        l[i] = 2*(xs[i+1]-xs[i-1]) - h[i-1]*mu[i-1]
        if l[i] == 0: l[i] = 1e-12
        mu[i] = h[i] / l[i]
        z[i]  = (alpha[i] - h[i-1]*z[i-1]) / l[i]
    l[n] = 1.0
    c = [0.0]*(n+1); b = [0.0]*n; dd = [0.0]*n
    for j in range(n-1, -1, -1):
        c[j]  = z[j] - mu[j]*c[j+1]
        b[j]  = (ys[j+1]-ys[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        dd[j] = (c[j+1]-c[j]) / (3*h[j]) if h[j] else 0
    return xs, [(ys[i], b[i], c[i], dd[i]) for i in range(n)]

def _eval_spline(x, xs, coeffs):
    """Evaluate cubic spline. Linear extrapolation beyond knot range."""
    n = len(xs) - 1
    if not coeffs: return 0.6
    if x <= xs[0]:
        a, b, c, d = coeffs[0];  return a + b*(x - xs[0])
    if x >= xs[-1]:
        a, b, c, d = coeffs[-1]
        dx = xs[-1] - xs[-2] if n > 0 else 0
        val   = a + b*dx + c*dx**2 + d*dx**3
        slope = b + 2*c*dx + 3*d*dx**2
        return val + slope*(x - xs[-1])
    lo_i, hi_i = 0, n-1
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        if xs[mid+1] < x: lo_i = mid+1
        else: hi_i = mid
    a, b, c, d = coeffs[lo_i];  dx = x - xs[lo_i]
    return a + b*dx + c*dx**2 + d*dx**3

def eval_spline_deriv(K, xs, coeffs):
    """
    Exact first derivative dσ/dK of the cubic IV spline at strike K.
    Tails: constant slope equal to the boundary tangent (linear extrapolation).
    """
    n = len(xs) - 1
    if not coeffs: return 0.0
    if K <= xs[0]:
        _, b, _, _ = coeffs[0];  return b          # left-tail constant slope
    if K >= xs[-1]:
        _, b, c, d = coeffs[-1]
        dx = (xs[-1] - xs[-2]) if n > 0 else 0.0
        return b + 2*c*dx + 3*d*dx**2               # right-tail constant slope
    lo_i, hi_i = 0, n-1
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        if xs[mid+1] < K: lo_i = mid+1
        else: hi_i = mid
    _, b, c, d = coeffs[lo_i];  dx = K - xs[lo_i]
    return b + 2*c*dx + 3*d*dx**2

def build_smile(options_sorted):
    """Pre-compute cubic spline from sorted options list. Returns (xs, coeffs)."""
    if len(options_sorted) < 2: return None
    xs = [o["strike"] for o in options_sorted]
    ys = [o["iv"] / 100 for o in options_sorted]
    return _build_spline(xs, ys)

def interp_iv(K, options_sorted, smile=None):
    """Interpolate IV at K. Returns decimal (e.g. 0.55)."""
    if not options_sorted: return 0.6
    if smile is not None:
        xs, coeffs = smile
        if coeffs: return max(_eval_spline(K, xs, coeffs), 0.05)
    if K <= options_sorted[0]["strike"]:  return options_sorted[0]["iv"] / 100
    if K >= options_sorted[-1]["strike"]: return options_sorted[-1]["iv"] / 100
    for i in range(len(options_sorted)-1):
        lo, hi = options_sorted[i], options_sorted[i+1]
        if lo["strike"] <= K <= hi["strike"]:
            t = (K - lo["strike"]) / (hi["strike"] - lo["strike"])
            return ((1-t)*lo["iv"] + t*hi["iv"]) / 100
    return options_sorted[0]["iv"] / 100

# ── Analytical Breeden-Litzenberger ──────────────────────────────────────────
def dCdK_smile(S, K, T, smile, r=0.0):
    """
    Exact ∂C/∂K via chain rule for C = BS(S, K, T, σ(K), r):
        ∂C/∂K = −e^{−rT}·N(d₂) + S·√T·φ(d₁)·σ'(K)
    where σ'(K) is the exact cubic spline derivative.
    """
    xs, coeffs = smile
    sigma      = max(_eval_spline(K, xs, coeffs), 0.05)
    dsigma_dK  = eval_spline_deriv(K, xs, coeffs)
    if T <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    return (-math.exp(-r * T) * ncdf(d2)
            + S * math.sqrt(T) * npdf(d1) * dsigma_dK)

def bl_bracket_prob(S, lo, hi, T, r, smile):
    """
    Model-free bracket probability via Breeden-Litzenberger with exact
    analytical spline derivatives. No finite-difference approximation.

        P(K1 < S_T < K2) = e^{rT} · [∂C/∂K(K2) − ∂C/∂K(K1)]
        P(S_T > K)       = −e^{rT} · ∂C/∂K(K)
        P(S_T < K)       = 1 + e^{rT} · ∂C/∂K(K)

    smile = (xs, coeffs) tuple from _build_spline / build_smile.
    """
    if smile is None: return None
    def dCdK(K): return dCdK_smile(S, K, T, smile, r)
    erT = math.exp(r * T)
    if lo is not None and hi is not None:
        return max(erT * (dCdK(hi) - dCdK(lo)), 0.0)
    elif lo is not None:
        return max(min(-erT * dCdK(lo), 1.0), 0.0)
    else:
        return max(min(1.0 + erT * dCdK(hi), 1.0), 0.0)

def check_smile_noarb(S, T, r, smile, n_points=40):
    """
    Check risk-neutral density for butterfly violations.
    Evaluates ∂C/∂K at a fine grid and flags zones where the slope is
    decreasing (i.e., d²C/dK² < 0, equivalent to negative risk-neutral PDF).
    Returns list of (K_lo, K_hi) violation intervals, empty if clean.
    """
    if not smile: return []
    xs, _ = smile
    K_min = xs[0]  * 0.85
    K_max = xs[-1] * 1.15
    step  = (K_max - K_min) / n_points
    Ks    = [K_min + i * step for i in range(n_points + 1)]
    dCdKs = [dCdK_smile(S, K, T, smile, r) for K in Ks]
    violations = []
    for i in range(1, len(dCdKs)):
        # dC/dK must be monotonically increasing (less negative) as K rises
        if dCdKs[i] < dCdKs[i-1] - 1e-6:
            violations.append(round(Ks[i]))
    return violations

# ── Temporal Variance Interpolation ──────────────────────────────────────────
def interpolate_iv_temporal(K, T_target, near_data, far_data):
    """
    Interpolate IV at strike K for T_target using total-variance interpolation:
        σ²(K,T_target)·T_target = lerp(σ²(K,T_near)·T_near, σ²(K,T_far)·T_far)
    """
    T_near = near_data['T_years']; T_far = far_data['T_years']
    iv_n   = interp_iv(K, near_data['options'], near_data['smile'])
    iv_f   = interp_iv(K, far_data['options'],  far_data['smile'])
    if T_far <= T_near or T_target <= 0: return iv_n
    if T_target <= T_near: return iv_n
    if T_target >= T_far:  return iv_f
    w      = (T_target - T_near) / (T_far - T_near)
    tv     = iv_n**2 * T_near * (1 - w) + iv_f**2 * T_far * w
    return math.sqrt(max(tv / T_target, 0.0025))  # floor 5%

def build_temporal_smile(T_target, near_data, far_data):
    """
    Build a synthetic cubic spline smile for T_target by evaluating
    interpolate_iv_temporal at the union of both expiries' knot points.
    Returns (xs, coeffs) ready for dCdK_smile / bl_bracket_prob.
    """
    near_smile = near_data.get('smile')
    far_smile  = far_data.get('smile')
    near_xs    = list(near_smile[0]) if near_smile else []
    far_xs     = list(far_smile[0])  if far_smile  else []
    all_ks     = sorted(set(near_xs + far_xs))
    if len(all_ks) < 2: return None
    ys = [max(interpolate_iv_temporal(K, T_target, near_data, far_data), 0.05)
          for K in all_ks]
    return _build_spline(all_ks, ys)

# ── Deribit ───────────────────────────────────────────────────────────────────
def fetch_deribit():
    """Fetch all BTC options. Returns (spot, by_expiry) with combined OTM smiles."""
    resp = _req.get(DERIBIT_URL, params={"currency": "BTC", "kind": "option"}, timeout=15)
    resp.raise_for_status()
    now = datetime.now(timezone.utc)
    raw = {}; spot = 0

    for item in resp.json().get("result", []):
        name  = item.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) != 4 or parts[3] not in ("C", "P"): continue
        try:
            strike    = float(parts[2])
            expiry_dt = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc, hour=8)
        except ValueError: continue
        days = (expiry_dt - now).days
        if days < 0 or days > 120: continue
        oi = float(item.get("open_interest") or 0)
        if oi < MIN_OI: continue
        iv = float(item.get("mark_iv") or 0)
        S  = float(item.get("underlying_price") or 0)
        if iv <= 0 or S <= 0: continue
        if S > spot: spot = S
        key = expiry_dt.strftime("%Y-%m-%d")
        if key not in raw:
            raw[key] = {"expiry": key, "expiry_dt": expiry_dt,
                        "calls": [], "puts": [], "spot": S}
        entry = {"instrument": name, "strike": strike, "iv": iv,
                 "oi": oi, "underlying": S, "type": parts[3]}
        raw[key]["calls" if parts[3] == "C" else "puts"].append(entry)

    by_expiry = {}
    for key, data in raw.items():
        S_exp = data.get("spot", spot)
        calls = sorted(data["calls"], key=lambda c: c["strike"])
        puts  = sorted(data["puts"],  key=lambda c: c["strike"])
        # OTM smile: puts below ATM, calls above ATM
        combined = {}
        for p in puts:
            if p["strike"] <= S_exp: combined[p["strike"]] = p
        for c in calls:
            K = c["strike"]
            if K > S_exp:
                combined[K] = c
            elif K not in combined:
                combined[K] = c
            elif K == S_exp and c["oi"] > combined[K]["oi"]:
                combined[K] = c
        options = sorted(combined.values(), key=lambda o: o["strike"])
        by_expiry[key] = {
            "expiry": key, "expiry_dt": data["expiry_dt"],
            "calls": calls, "puts": puts, "options": options,
        }
    return spot, by_expiry

def fetch_coinbase_spot():
    try:
        r = _req.get(COINBASE_URL, timeout=8)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception: return None

def fetch_realized_vol():
    try:
        r = _req.get(DERIBIT_HV, params={"currency": "BTC"}, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", [])
        if data: return data[-1][1] / 100
    except Exception: pass
    return None

def compute_atm_iv(spot, by_expiry):
    if not by_expiry: return None
    now = datetime.now(timezone.utc)
    nearest_key = None; nearest_days = float("inf")
    for k, v in by_expiry.items():
        days = (v["expiry_dt"] - now).total_seconds() / 86400
        opts = v.get("options", v.get("calls", []))
        if 0 < days < nearest_days and len(opts) >= 3:
            nearest_days = days; nearest_key = k
    if not nearest_key: return None
    options = by_expiry[nearest_key].get("options", by_expiry[nearest_key].get("calls", []))
    best_dist = float("inf"); atm_iv = None
    for o in options:
        dist = abs(o["strike"] - spot)
        if dist < best_dist: best_dist = dist; atm_iv = o["iv"] / 100
    return atm_iv

# ── Polymarket CLOB ───────────────────────────────────────────────────────────
_clob_session = None
def _get_clob_session():
    global _clob_session
    if _clob_session is None:
        _clob_session = _req.Session()
        _clob_session.headers.update({"User-Agent": "btc-dashboard/2.0"})
    return _clob_session

def fetch_clob_mid_spread(token_id):
    """Fetch live CLOB midpoint and spread for a YES token_id.
    Returns (mid_price, spread) in [0,1], or (None, None) on failure."""
    if not token_id: return None, None
    s = _get_clob_session()
    try:
        mid_r = s.get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id}, timeout=5)
        md    = mid_r.json()
        mid   = float(md.get("mid") or md.get("mid_price") or md.get("midpoint") or 0)
        spr_r = s.get(f"{CLOB_BASE}/spread",   params={"token_id": token_id}, timeout=5)
        sd    = spr_r.json()
        spread = float(sd.get("spread") or 0)
        if 0 < mid < 1: return mid, spread
    except Exception: pass
    return None, None

def fetch_clob_batch(token_ids):
    """Fetch CLOB mid+spread for multiple token IDs in parallel.
    Returns {token_id: (mid, spread)}."""
    results = {}
    if not token_ids: return results
    def _one(tid): return tid, fetch_clob_mid_spread(tid)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(_one, tid): tid for tid in token_ids if tid}
        for fut in as_completed(futs, timeout=12):
            try:
                tid, val = fut.result()
                results[tid] = val
            except Exception: pass
    return results

# ── Polymarket events ─────────────────────────────────────────────────────────
MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]

def candidate_slugs():
    now = datetime.now(timezone.utc); slugs = []
    for delta in range(-1, 90):
        d = now + timedelta(days=delta)
        slugs.append(f"bitcoin-price-on-{MONTHS[d.month-1]}-{d.day}")
    return slugs

def fetch_poly_events():
    events = []; seen = set()
    for slug in candidate_slugs():
        try:
            r     = _req.get(POLY_URL, params={"slug": slug}, timeout=8)
            batch = r.json() if isinstance(r.json(), list) else []
            for ev in batch:
                eid = ev.get("id","")
                if ev.get("closed") or eid in seen: continue
                markets = ev.get("markets", [])
                if not any("price" in (m.get("question","")).lower() and
                           any(w in m.get("question","").lower()
                               for w in ("between","above","below","under"))
                           for m in markets): continue
                seen.add(eid); events.append(ev)
        except Exception: continue
    if not events:
        try:
            r = _req.get(POLY_MKTS, params={"closed":"false","tag":"crypto","limit":100}, timeout=10)
            mkts = r.json() if isinstance(r.json(), list) else []
            ev_map = {}
            for m in mkts:
                q = m.get("question","")
                if not ("bitcoin" in q.lower() or "btc" in q.lower()): continue
                if not any(w in q.lower() for w in ("between","above","below","price")): continue
                gid = m.get("groupItemTitle") or m.get("slug","fallback")
                if gid not in ev_map:
                    ev_map[gid] = {"id":gid,"title":gid,"slug":gid,
                                   "closed":False,"endDate":m.get("endDate",""),"markets":[]}
                ev_map[gid]["markets"].append(m)
            for ev in ev_map.values():
                if ev["markets"]: events.append(ev)
        except Exception: pass
    return events

def parse_all_brackets(question):
    amounts = []
    for m in re.finditer(r'\$([\d,]+)', question):
        v = float(m.group(1).replace(",",""))
        if v > 1000: amounts.append(v)
    q = question.lower()
    if "between" in q and len(amounts) >= 2: return sorted(amounts[:2])[0], sorted(amounts[:2])[1]
    elif "above" in q and len(amounts) >= 1: return max(amounts), None
    elif ("below" in q or "under" in q) and len(amounts) >= 1: return None, min(amounts)
    return None, None

# ── Core computation ──────────────────────────────────────────────────────────
def compute_data():
    t0 = time.time()
    deribit_spot, deribit = fetch_deribit()
    poly_events  = fetch_poly_events()
    now          = datetime.now(timezone.utc)

    coinbase_spot = fetch_coinbase_spot()
    spot  = coinbase_spot if coinbase_spot else deribit_spot
    basis = round(deribit_spot - coinbase_spot, 2) if coinbase_spot else None

    atm_iv       = compute_atm_iv(spot, deribit)
    realized_vol = fetch_realized_vol()
    r            = RISK_FREE

    # Pre-compute T_years and smiles for each Deribit expiry
    expiry_data = {}
    for key, v in deribit.items():
        T       = max((v["expiry_dt"] - now).total_seconds() / (365.25 * 86400), 1e-9)
        options = v.get("options", [])
        smile   = build_smile(options) if len(options) >= 2 else None
        expiry_data[key] = {"T_years": T, "options": options,
                            "smile": smile, "expiry_dt": v["expiry_dt"]}

    result_events = []

    for ev in poly_events:
        end_raw = ev.get("endDate","")
        try: end_dt = datetime.fromisoformat(end_raw.replace("Z","+00:00"))
        except Exception: continue

        T_target = max((end_dt - now).total_seconds() / (365.25 * 86400), 1e-9)

        # Find bracketing Deribit expiries for temporal interpolation
        sorted_keys = sorted(expiry_data, key=lambda k: expiry_data[k]["T_years"])
        near_key = far_key = best_key = None
        best_diff = float("inf")
        for k in sorted_keys:
            diff = abs(expiry_data[k]["T_years"] - T_target)
            if diff < best_diff: best_diff = diff; best_key = k
        for k in sorted_keys:
            if expiry_data[k]["T_years"] <= T_target: near_key = k
        for k in sorted_keys:
            if expiry_data[k]["T_years"] >= T_target: far_key = k; break

        use_interp = (near_key and far_key and near_key != far_key
                      and len(expiry_data[near_key]["options"]) >= 2
                      and len(expiry_data[far_key]["options"])  >= 2)

        if use_interp:
            working_smile  = build_temporal_smile(T_target, expiry_data[near_key], expiry_data[far_key])
            interp_method  = "variance_interp"
            matched_expiry = f"{near_key} ↔ {far_key}"
            ref_d          = expiry_data[near_key]
        else:
            if not best_key or not expiry_data[best_key]["options"]: continue
            working_smile  = expiry_data[best_key]["smile"]
            interp_method  = "nearest"
            matched_expiry = best_key
            ref_d          = expiry_data[best_key]

        if not working_smile: continue

        # No-arbitrage check on the working smile
        arb_violations = check_smile_noarb(spot, T_target, r, working_smile)

        # ── Collect YES token IDs for batch CLOB fetch ────────────────────
        token_map = {}   # market_idx -> yes_token_id
        for idx, m in enumerate(ev.get("markets", [])):
            clob_ids = m.get("clobTokenIds") or []
            if isinstance(clob_ids, str):
                try: clob_ids = json.loads(clob_ids)
                except: clob_ids = []
            if not clob_ids:
                tokens = m.get("tokens") or []
                for t in (tokens if isinstance(tokens, list) else []):
                    if isinstance(t, dict) and t.get("outcome","").lower() == "yes":
                        clob_ids = [t.get("token_id") or t.get("tokenId","")]
                        break
            if clob_ids and clob_ids[0]:
                token_map[idx] = clob_ids[0]

        clob_results = fetch_clob_batch(list(token_map.values()))

        # ── Build raw brackets ─────────────────────────────────────────────
        raw_brackets = []
        for idx, m in enumerate(ev.get("markets", [])):
            q    = m.get("question","")
            lo, hi = parse_all_brackets(q)
            if lo is None and hi is None: continue

            # Polymarket price: CLOB midpoint preferred, fallback to outcomePrices
            poly_raw = 0.5; poly_spread = None; poly_source = "gamma"
            tid = token_map.get(idx)
            if tid and tid in clob_results:
                mid, spread = clob_results[tid]
                if mid is not None:
                    poly_raw = mid; poly_spread = spread; poly_source = "clob"
            if poly_source == "gamma":
                raw_p = m.get("outcomePrices","")
                try:
                    prices = json.loads(raw_p) if isinstance(raw_p, str) else raw_p
                    poly_raw = float(prices[0])
                except Exception: pass

            # Breeden-Litzenberger probability (analytical, model-free)
            try:
                opt_prob_bl = bl_bracket_prob(spot, lo, hi, T_target, r, working_smile)
            except Exception:
                opt_prob_bl = None

            # N(d₂) reference (for display comparison)
            try:
                ref_opts  = ref_d["options"]; ref_smile = ref_d["smile"]
                if lo is not None and hi is not None:
                    iv_lo = interp_iv(lo, ref_opts, ref_smile)
                    iv_hi = interp_iv(hi, ref_opts, ref_smile)
                    opt_nd2 = max(prob_above(spot,lo,T_target,iv_lo,r) - prob_above(spot,hi,T_target,iv_hi,r), 0.0)
                elif lo is not None:
                    opt_nd2 = prob_above(spot, lo, T_target, interp_iv(lo, ref_opts, ref_smile), r)
                else:
                    opt_nd2 = 1.0 - prob_above(spot, hi, T_target, interp_iv(hi, ref_opts, ref_smile), r)
            except Exception: opt_nd2 = None

            label = (f"${lo/1000:.0f}k–${hi/1000:.0f}k" if lo is not None and hi is not None
                     else f">${lo/1000:.0f}k" if lo is not None
                     else f"<${hi/1000:.0f}k")
            mid_val = ((lo+hi)/2 if lo is not None and hi is not None
                       else lo*1.05 if lo is not None else hi*0.95)

            raw_brackets.append({
                "label": label, "lo": lo, "hi": hi, "mid": mid_val,
                "poly_raw": poly_raw, "poly_spread": poly_spread,
                "poly_source": poly_source,
                "opt_bl": opt_prob_bl, "opt_nd2": opt_nd2,
                "question": q[:80], "volume": float(m.get("volume") or 0),
                "is_tail": False,
            })

        raw_brackets.sort(key=lambda b: b["mid"])

        # ── Add tail probability rows (B-L only, no Polymarket price) ─────
        all_los = [b["lo"] for b in raw_brackets if b["lo"] is not None]
        all_his = [b["hi"] for b in raw_brackets if b["hi"] is not None]

        if all_his:
            min_lo_of_bounds = min(b["lo"] for b in raw_brackets if b["lo"] is not None) if all_los else None
            min_hi = min(all_his)
            # Left tail: P(S_T < min(upper_bounds))
            tail_lo = min(min_hi, min_lo_of_bounds) if min_lo_of_bounds else min_hi
            try:
                p_left = bl_bracket_prob(spot, None, tail_lo, T_target, r, working_smile)
            except Exception: p_left = None
            if p_left is not None and p_left > 0.002:
                raw_brackets.insert(0, {
                    "label": f"<${tail_lo/1000:.0f}k (tail)",
                    "lo": None, "hi": tail_lo, "mid": tail_lo * 0.88,
                    "poly_raw": None, "poly_spread": None, "poly_source": "tail",
                    "opt_bl": p_left, "opt_nd2": None,
                    "question": "", "volume": 0, "is_tail": True,
                })

        if all_los:
            max_lo = max(all_los)
            try:
                p_right = bl_bracket_prob(spot, max_lo, None, T_target, r, working_smile)
            except Exception: p_right = None
            if p_right is not None and p_right > 0.002:
                raw_brackets.append({
                    "label": f">${max_lo/1000:.0f}k (tail)",
                    "lo": max_lo, "hi": None, "mid": max_lo * 1.12,
                    "poly_raw": None, "poly_spread": None, "poly_source": "tail",
                    "opt_bl": p_right, "opt_nd2": None,
                    "question": "", "volume": 0, "is_tail": True,
                })

        # ── Normalize Polymarket overround (non-tail brackets only) ────────
        poly_sum = sum(b["poly_raw"] for b in raw_brackets
                       if b["poly_raw"] is not None and not b["is_tail"])
        overround = poly_sum

        brackets = []
        for b in raw_brackets:
            if b["is_tail"]:
                poly_norm = None
            elif poly_sum > 0:
                poly_norm = b["poly_raw"] / poly_sum
            else:
                poly_norm = b["poly_raw"]

            opt_bl  = b["opt_bl"]
            edge    = (opt_bl - poly_norm) if (opt_bl is not None and poly_norm is not None) else None
            # Half-spread uncertainty on Polymarket side
            half_spread = (b["poly_spread"] / 2) if b["poly_spread"] else None

            brackets.append({
                "label":        b["label"],
                "lo":           b["lo"],
                "hi":           b["hi"],
                "mid":          b["mid"],
                "poly_raw":     round(b["poly_raw"] * 100, 2) if b["poly_raw"] is not None else None,
                "poly_prob":    round(poly_norm * 100, 2) if poly_norm is not None else None,
                "poly_spread":  round(b["poly_spread"] * 100, 2) if b["poly_spread"] is not None else None,
                "poly_source":  b["poly_source"],
                "opt_prob":     round(opt_bl  * 100, 2) if opt_bl  is not None else None,
                "opt_nd2":      round(b["opt_nd2"] * 100, 2) if b["opt_nd2"] is not None else None,
                "edge":         round(edge * 100, 2) if edge is not None else None,
                "half_spread":  round(half_spread * 100, 2) if half_spread is not None else None,
                "is_tail":      b["is_tail"],
                "question":     b["question"],
                "volume":       b["volume"],
            })

        if brackets:
            result_events.append({
                "title":          ev.get("title",""),
                "slug":           ev.get("slug",""),
                "end_date":       end_raw,
                "end_label":      end_dt.strftime("%b %-d, %H:%M UTC"),
                "deribit_expiry": matched_expiry,
                "interp_method":  interp_method,
                "brackets":       brackets,
                "overround":      round(overround * 100, 1),
                "total_poly":     round(sum(b["poly_prob"] for b in brackets if b["poly_prob"] is not None), 1),
                "total_opt":      round(sum(b["opt_prob"]  for b in brackets if b["opt_prob"]  is not None), 1),
                "arb_violations": len(arb_violations),
                "clob_brackets":  sum(1 for b in brackets if b["poly_source"] == "clob"),
            })

    # ── Vol surface ───────────────────────────────────────────────────────────
    vol_surface = []
    for key in sorted(deribit):
        exp_data = deribit[key]
        for c in exp_data.get("calls", []):
            T     = max((exp_data["expiry_dt"] - now).total_seconds() / (365.25*86400), 1e-9)
            sigma = c["iv"] / 100
            delta_c = ncdf(bs_d1(spot, c["strike"], T, sigma, r))
            p_itm   = ncdf(bs_d2(spot, c["strike"], T, sigma, r))
            vol_surface.append({
                "instrument": c["instrument"], "expiry": key,
                "strike": c["strike"], "iv": c["iv"],
                "delta": round(delta_c, 3), "prob_itm": round(p_itm, 4), "oi": c["oi"],
            })

    return {
        "spot":          spot,
        "deribit_spot":  deribit_spot,
        "coinbase_spot": coinbase_spot,
        "basis":         basis,
        "updated":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "elapsed":       round(time.time() - t0, 1),
        "events":        result_events,
        "vol_surface":   vol_surface,
        "n_deribit":     len(vol_surface),
        "n_events":      len(result_events),
        "atm_iv":        round(atm_iv * 100, 1) if atm_iv else None,
        "realized_vol":  round(realized_vol * 100, 1) if realized_vol else None,
        "risk_free":     round(r * 100, 2),
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
*{box-sizing:border-box;margin:0;padding:0}
body{background:#eef2f7;color:#1e293b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:15px;min-height:100vh;line-height:1.5}
.info-tip{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:#dbeafe;color:#2563eb;font-size:11px;font-style:italic;font-weight:800;cursor:help;vertical-align:middle;margin-left:5px;flex-shrink:0;user-select:none;border:1.5px solid #bfdbfe}
#jtip{position:fixed;z-index:99999;background:#1e293b;color:#f1f5f9;padding:12px 15px;border-radius:9px;font-size:13px;font-weight:400;font-style:normal;max-width:310px;width:max-content;line-height:1.6;text-align:left;pointer-events:none;opacity:0;transition:opacity .15s;box-shadow:0 6px 24px rgba(0,0,0,.22);word-break:normal}
.topbar{background:#ffffff;border-bottom:2px solid #e2e8f0;padding:14px 28px;display:flex;align-items:center;gap:22px;flex-wrap:wrap;position:sticky;top:0;z-index:100;box-shadow:0 2px 10px rgba(0,0,0,.07)}
.topbar h1{font-size:20px;font-weight:800;color:#0f172a;letter-spacing:-.01em;white-space:nowrap}
.stat{display:flex;flex-direction:column;gap:2px}
.stat label{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.09em;font-weight:600;display:flex;align-items:center}
.stat span{font-size:17px;font-weight:700;color:#0f172a}
#countdown{margin-left:auto;font-size:14px;color:#94a3b8;white-space:nowrap;display:flex;align-items:center;gap:7px}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;display:inline-block;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
.main{padding:22px 28px;display:grid;gap:20px}
.card{background:#ffffff;border:1.5px solid #e2e8f0;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
.card-hdr{padding:14px 18px;border-bottom:1.5px solid #f1f5f9;display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#f8fafc;border-radius:12px 12px 0 0}
.card-hdr h2{font-size:14px;text-transform:uppercase;letter-spacing:.07em;color:#475569;font-weight:800}
.event-tabs{display:flex;gap:9px;flex-wrap:wrap;padding:14px 18px;border-bottom:1.5px solid #f1f5f9;background:#fafbfd}
.etab{padding:7px 18px;border-radius:22px;font-size:14px;font-weight:600;cursor:pointer;border:1.5px solid #e2e8f0;color:#64748b;background:#fff;transition:all .15s}
.etab:hover{border-color:#93c5fd;color:#2563eb;background:#eff6ff}
.etab.active{background:#2563eb;border-color:#2563eb;color:#ffffff;box-shadow:0 2px 8px rgba(37,99,235,.3)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.chart-wrap{padding:18px;height:400px;position:relative}
.chart-wrap-sm{padding:18px;height:310px;position:relative}
table{width:100%;border-collapse:collapse}
th{padding:12px 16px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#475569;border-bottom:2px solid #e2e8f0;text-align:right;background:#f8fafc;font-weight:700;white-space:nowrap}
th:first-child{text-align:left}
td{padding:10px 16px;border-bottom:1px solid #f1f5f9;font-size:14px;text-align:right;font-weight:500}
td:first-child{text-align:left}
tr:last-child td{border:none}
tr:hover td{background:#f8fafc}
tr.tail-row td{background:#fafbfd;color:#94a3b8;font-style:italic}
tr.tail-row:hover td{background:#f1f5f9}
.under{color:#16a34a;font-weight:700}.over{color:#dc2626;font-weight:700}.fair{color:#94a3b8;font-weight:600}
.pill{display:inline-block;padding:4px 12px;border-radius:5px;font-size:13px;font-weight:700;letter-spacing:.03em}
.pill-under{background:#dcfce7;color:#16a34a}.pill-over{background:#fee2e2;color:#dc2626}.pill-fair{background:#f1f5f9;color:#94a3b8}
.arb-flag{display:inline-flex;align-items:center;gap:5px;background:#fff7ed;border:1.5px solid #fed7aa;border-radius:6px;padding:3px 10px;font-size:12px;font-weight:700;color:#c2410c}
.clob-badge{display:inline-block;background:#dcfce7;color:#15803d;font-size:10px;font-weight:800;border-radius:3px;padding:1px 5px;vertical-align:middle;margin-left:4px}
.gamma-badge{display:inline-block;background:#fef9c3;color:#854d0e;font-size:10px;font-weight:800;border-radius:3px;padding:1px 5px;vertical-align:middle;margin-left:4px}
.legend-bar{display:flex;align-items:center;gap:22px;flex-wrap:wrap;padding:12px 18px;border-top:1.5px solid #f1f5f9;background:#fafbfd;font-size:14px;font-weight:600}
.legend-item{display:flex;align-items:center;gap:8px}
.legend-swatch{width:24px;height:15px;border-radius:4px;flex-shrink:0}
.legend-line-swatch{width:28px;height:3px;border-radius:2px;flex-shrink:0}
.totals{display:flex;gap:24px;padding:12px 18px;background:#f8fafc;font-size:14px;color:#64748b;border-top:1.5px solid #f1f5f9;flex-wrap:wrap;align-items:center}
.totals b{color:#0f172a;font-size:16px}
#loading{position:fixed;inset:0;background:#eef2f7;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:14px;z-index:200}
.spin{width:40px;height:40px;border:3px solid #e2e8f0;border-top-color:#2563eb;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{padding:36px;text-align:center;color:#94a3b8;font-size:15px}
.insight-box{background:#fefce8;border:1.5px solid #fde68a;border-radius:9px;padding:13px 17px;margin:14px 18px 0 18px;font-size:13px;color:#78350f;line-height:1.65}
.insight-box strong{color:#92400e;font-size:14px}
.method-box{background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:9px;padding:13px 17px;margin:0 0 4px 0;font-size:13px;color:#14532d;line-height:1.65}
.method-box strong{color:#166534;font-size:14px}
.no-events-notice{background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:9px;padding:16px 20px;margin:16px 18px;font-size:14px;color:#1e40af;line-height:1.7}
.no-events-notice strong{color:#1d4ed8}
.spread-band{font-size:11px;color:#94a3b8;font-weight:500}
</style>
</head>
<body>
<div id="jtip"></div>
<div id="loading"><div class="spin"></div><div style="color:#64748b;font-size:15px;margin-top:6px">Fetching Deribit + Polymarket CLOB + Coinbase&#8230;</div></div>

<div class="topbar">
  <h1>&#9889; BTC Options vs Polymarket</h1>
  <div class="stat">
    <label>Coinbase Spot <span class="info-tip" data-tip="BTC spot from Coinbase — Polymarket's settlement reference. Used as S in all calculations.">i</span></label>
    <span id="spot">&#8212;</span>
  </div>
  <div class="stat">
    <label>Deribit Index <span class="info-tip" data-tip="Deribit 8-exchange BTC index. Options are priced against this.">i</span></label>
    <span id="deribit-spot">&#8212;</span>
  </div>
  <div class="stat">
    <label>Basis <span class="info-tip" data-tip="Deribit index minus Coinbase spot. A source of settlement mismatch between the two markets.">i</span></label>
    <span id="basis">&#8212;</span>
  </div>
  <div class="stat">
    <label>ATM IV <span class="info-tip" data-tip="At-the-money implied vol from nearest Deribit expiry. Compare to Realized Vol to see the variance risk premium.">i</span></label>
    <span id="atm-iv">&#8212;</span>
  </div>
  <div class="stat">
    <label>Realized Vol <span class="info-tip" data-tip="BTC historical volatility from Deribit.">i</span></label>
    <span id="realized-vol">&#8212;</span>
  </div>
  <div class="stat">
    <label>Contracts <span class="info-tip" data-tip="Deribit option contracts loaded (OI≥3, expiry≤120d). Includes calls + puts for combined smile.">i</span></label>
    <span id="nd">&#8212;</span>
  </div>
  <div class="stat">
    <label>Events <span class="info-tip" data-tip="Active Polymarket BTC price-bracket events.">i</span></label>
    <span id="ne">&#8212;</span>
  </div>
  <div class="stat">
    <label>Updated</label>
    <span id="upd" style="font-size:13px;color:#94a3b8;font-weight:400">&#8212;</span>
  </div>
  <div id="countdown"><span class="dot"></span>next refresh in <b id="ct" style="color:#0f172a;font-size:15px">&#8212;</b>s</div>
</div>

<div class="main">

  <div class="method-box">
    <strong>Methodology v2.0 — Arbitrage-free &amp; statistically rigorous:</strong>
    Both columns are <em>risk-neutral</em> probabilities from independent markets.
    <strong>Options (B-L):</strong> Breeden-Litzenberger model-free extraction using exact analytical spline derivatives
    ∂C/∂K = −e<sup>−rT</sup>N(d₂) + vega·σ'(K) — no finite-difference step, no log-normal assumption.
    Combined OTM put+call smile. Temporal variance interpolation for expiry mismatch.
    Tail rows show probability mass outside Polymarket bracket coverage.
    Butterfly violations flagged &#9888;.
    <strong>Polymarket:</strong> Live CLOB midpoints with ±spread bands <span class="clob-badge">CLOB</span>,
    normalized for ~2–5% overround. Basis: Coinbase vs. Deribit shown explicitly.
    r = <span id="rfr">4.3</span>%.
  </div>

  <div class="card">
    <div class="card-hdr">
      <h2>Risk-Neutral Probability Comparison</h2>
      <span class="info-tip" data-tip="Both columns are risk-neutral probabilities. Options: Breeden-Litzenberger with exact analytical derivatives on combined put+call smile. Polymarket: live CLOB midpoints normalized for overround. Edge is a genuine market disagreement.">i</span>
      <span id="arb-flag" style="display:none;margin-left:8px"></span>
      <span id="event-meta" style="font-size:13px;color:#94a3b8;margin-left:auto"></span>
    </div>

    <div class="event-tabs" id="event-tabs">
      <span style="color:#94a3b8;font-size:14px;padding:4px">Loading events&#8230;</span>
    </div>

    <div id="no-events-panel" style="display:none">
      <div class="no-events-notice">
        <strong>No active Polymarket BTC price events found.</strong><br>
        Today's event has likely settled. New events appear a few hours before midnight UTC.
        The vol surface below is still live.
      </div>
    </div>

    <div class="chart-wrap" id="main-chart-wrap"><canvas id="mainChart"></canvas></div>

    <div class="legend-bar">
      <div class="legend-item">
        <div class="legend-swatch" style="background:rgba(37,99,235,0.70);border:1.5px solid #2563eb"></div>
        <span style="color:#2563eb">Polymarket (normalized)</span>
        <span class="info-tip" data-tip="Live CLOB midpoint (when available) or Gamma mark price, divided by sum to remove ~2-5% overround. ±spread band shown in Edge column for CLOB-sourced prices.">i</span>
      </div>
      <div class="legend-item">
        <div class="legend-line-swatch" style="background:#7c3aed"></div>
        <span style="color:#7c3aed">Options B-L (risk-neutral)</span>
        <span class="info-tip" data-tip="Breeden-Litzenberger model-free probability using exact analytical spline derivatives. No log-normal assumption. Includes tail rows for probability mass outside the bracket coverage.">i</span>
      </div>
      <div style="margin-left:auto;font-size:13px;color:#94a3b8;font-weight:500">
        Edge = Options (B-L) &#8722; Polymarket (norm.) &nbsp;|&nbsp; ±band = CLOB half-spread
      </div>
    </div>

    <div class="totals">
      <div>Poly (norm.): <b id="poly-total">&#8212;</b>%
        <span class="info-tip" data-tip="Sum of normalized Polymarket probabilities across non-tail brackets. ~100% by construction.">i</span>
      </div>
      <div>Options (B-L): <b id="opt-total">&#8212;</b>%
        <span class="info-tip" data-tip="Sum of all B-L probabilities including tail rows. ~100% when tails are included.">i</span>
      </div>
      <div style="font-size:13px;color:#94a3b8">
        Poly overround: <b id="poly-overround" style="color:#0f172a">&#8212;</b>%
        <span class="info-tip" data-tip="Sum of raw Polymarket YES prices before normalization. The excess above 100% is the house edge.">i</span>
      </div>
    </div>

    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th style="text-align:left">Bracket
            <span class="info-tip" data-tip="BTC price range. Tail rows (italics) show probability mass outside Polymarket's bracket coverage — the options market sees this; Polymarket doesn't.">i</span>
          </th>
          <th><span style="color:#2563eb">Poly (norm.)</span>
            <span class="info-tip" data-tip="Normalized Polymarket YES price. CLOB badge = live order book midpoint. Gamma badge = last mark price (may be stale for illiquid brackets).">i</span>
          </th>
          <th><span style="color:#7c3aed">Options (B-L)</span>
            <span class="info-tip" data-tip="Breeden-Litzenberger model-free probability with exact analytical spline derivatives. Risk-neutral.">i</span>
          </th>
          <th>Edge ± Band
            <span class="info-tip" data-tip="Options (B-L) minus Polymarket (normalized). Band = ±CLOB half-spread. Only trade edges where |edge| > band (the band shows minimum Polymarket uncertainty). Both measures are risk-neutral so this is a genuine market disagreement.">i</span>
          </th>
          <th>Signal
            <span class="info-tip" data-tip="UNDER (green) = edge > +3pp. OVER (red) = edge < -3pp. FAIR = within ±3pp. Tail rows show no signal.">i</span>
          </th>
          <th>Vol ($)
            <span class="info-tip" data-tip="Polymarket USD volume. Higher = more reliable price.">i</span>
          </th>
        </tr></thead>
        <tbody id="bracket-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-hdr">
        <h2>Options Probability Smile</h2>
        <span class="info-tip" data-tip="N(d2) for each strike — risk-neutral P(BTC > K at expiry).">i</span>
        <span id="smile-expiry" style="font-size:13px;color:#94a3b8;margin-left:auto"></span>
      </div>
      <div class="insight-box">
        <strong>What this tells you:</strong> Where the curve crosses 50% is the market's expected settlement. Left asymmetry = crash risk priced in.
      </div>
      <div class="chart-wrap-sm"><canvas id="smileChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-hdr">
        <h2>Implied Volatility Smile</h2>
        <span class="info-tip" data-tip="Combined OTM put+call smile: puts below ATM, calls above ATM. The trough is the baseline vol forecast.">i</span>
        <span id="iv-expiry" style="font-size:13px;color:#94a3b8;margin-left:auto"></span>
      </div>
      <div class="insight-box">
        <strong>What this tells you:</strong> Steep left wing = expensive crash insurance. Trough vs. Realized Vol gap = variance risk premium.
      </div>
      <div class="chart-wrap-sm"><canvas id="ivChart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <div class="card-hdr">
      <h2>Deribit Vol Surface &#8212; Calls</h2>
      <span class="info-tip" data-tip="Risk-neutral values (standard for derivatives). P(ITM) = N(d2).">i</span>
      <select id="expiry-sel" style="background:#fff;color:#374151;border:1.5px solid #e2e8f0;border-radius:7px;padding:5px 12px;font-size:13px;cursor:pointer;margin-left:auto;font-weight:600">
        <option value="all">All expirations</option>
      </select>
    </div>
    <div style="overflow-x:auto;max-height:420px;overflow-y:auto">
      <table>
        <thead><tr>
          <th style="text-align:left">Instrument</th>
          <th>Strike</th><th>Expiry</th><th>Delta</th>
          <th>P(ITM)</th><th>IV</th><th>OI</th>
        </tr></thead>
        <tbody id="surface-body"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const REFRESH = REFRESH_PLACEHOLDER;
let countdown = REFRESH, allData = null, activeEventIdx = 0;
let mainChart, smileChart, ivChart;

(function(){
  const tip = document.getElementById('jtip');
  function pos(e){
    const pad=10,w=Math.min(tip.offsetWidth||310,310),h=tip.offsetHeight||60;
    let x=e.clientX-w/2,y=e.clientY-h-14;
    if(x<pad)x=pad; if(x+w>window.innerWidth-pad)x=window.innerWidth-w-pad;
    if(y<pad)y=e.clientY+22;
    tip.style.left=x+'px';tip.style.top=y+'px';
  }
  document.addEventListener('mouseover',e=>{
    const el=e.target.closest&&e.target.closest('.info-tip');
    if(el&&el.dataset.tip){tip.textContent=el.dataset.tip;tip.style.opacity='1';pos(e);}
  });
  document.addEventListener('mousemove',e=>{if(tip.style.opacity==='1')pos(e);});
  document.addEventListener('mouseout',e=>{if(e.target.closest&&e.target.closest('.info-tip'))tip.style.opacity='0';});
})();

const gc='#f1f5f9',tc='#64748b';
const baseOpts=(yLabel,xLabel)=>({
  responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',titleColor:'#f8fafc',bodyColor:'#cbd5e1',borderColor:'#334155',borderWidth:1,titleFont:{size:14,weight:'600'},bodyFont:{size:13},padding:12}},
  scales:{
    x:{ticks:{color:tc,font:{size:13},maxRotation:38},grid:{color:gc},title:{display:!!xLabel,text:xLabel||'',color:tc,font:{size:13,weight:'600'}}},
    y:{ticks:{color:tc,font:{size:13},callback:v=>v+'%'},grid:{color:gc},title:{display:true,text:yLabel,color:tc,font:{size:13,weight:'600'}}}
  }
});
function fmt$(n){return '$'+Number(n).toLocaleString('en-US',{maximumFractionDigits:0});}

function renderEvent(events,idx){
  if(!events||!events.length) return;
  const ev=events[idx];
  const interpLabel = ev.interp_method==='variance_interp'?' (variance interp.)':'';
  const clobLabel   = ev.clob_brackets>0?` · ${ev.clob_brackets} CLOB`:'';
  document.getElementById('event-meta').textContent=
    `expires ${ev.end_label}  ·  Deribit: ${ev.deribit_expiry}${interpLabel}${clobLabel}`;
  document.getElementById('poly-total').textContent=ev.total_poly;
  document.getElementById('opt-total').textContent=ev.total_opt||'—';
  document.getElementById('poly-overround').textContent=ev.overround||'—';

  // ARB flag
  const arbEl=document.getElementById('arb-flag');
  if(ev.arb_violations>0){
    arbEl.style.display='inline-flex';
    arbEl.innerHTML=`<span class="arb-flag">⚠ Smile: ${ev.arb_violations} arbitrage zone${ev.arb_violations>1?'s':''}</span>`;
  } else { arbEl.style.display='none'; }

  const bs=ev.brackets;
  // Chart: exclude tail rows (no poly_prob)
  const chartBs=bs.filter(b=>!b.is_tail);
  const labels   = chartBs.map(b=>b.label);
  const polyData = chartBs.map(b=>b.poly_prob);
  const optData  = chartBs.map(b=>b.opt_prob??null);

  const chartData={labels,datasets:[
    {label:'Polymarket (normalized)',data:polyData,type:'bar',
     backgroundColor:'rgba(37,99,235,0.70)',borderColor:'rgba(37,99,235,0.95)',
     borderWidth:1.5,borderRadius:5},
    {label:'Options B-L (risk-neutral)',data:optData,type:'line',
     backgroundColor:'transparent',borderColor:'rgba(124,58,237,0.95)',
     borderWidth:2.5,tension:0.35,pointRadius:5,
     pointBackgroundColor:'rgba(124,58,237,0.9)',pointHoverRadius:7,
     fill:false,yAxisID:'y'}
  ]};
  const chartOpts={...baseOpts('Probability %','Price bracket'),plugins:{
    tooltip:{backgroundColor:'#1e293b',titleColor:'#f8fafc',bodyColor:'#cbd5e1',borderColor:'#334155',borderWidth:1,titleFont:{size:14,weight:'600'},bodyFont:{size:13},padding:12,
      callbacks:{afterBody:(items)=>{
        const i=items[0].dataIndex,b=chartBs[i];
        if(b.edge!==null){
          const sign=b.edge>=0?'+':'';
          const band=b.half_spread?` ± ${b.half_spread.toFixed(1)}%`:'';
          return[`Edge: ${sign}${b.edge}${band}`,
            b.edge>3?'↑ Options sees HIGHER prob':b.edge<-3?'↓ Options sees LOWER prob':'~ Both crowds agree'];
        }
        return[];
      }}
    },
    legend:{display:true,position:'top',labels:{font:{size:14,weight:'600'},padding:18,usePointStyle:true,
      generateLabels:(chart)=>chart.data.datasets.map((ds,i)=>({
        text:ds.label,fillStyle:ds.type==='line'?'transparent':ds.backgroundColor,
        strokeStyle:ds.borderColor,lineWidth:ds.type==='line'?2.5:0,
        color:ds.borderColor,hidden:false,datasetIndex:i,
        pointStyle:ds.type==='line'?'line':'rect'
      }))
    }}
  }};
  if(mainChart){mainChart.data=chartData;mainChart.options=chartOpts;mainChart.update();}
  else mainChart=new Chart(document.getElementById('mainChart').getContext('2d'),{type:'bar',data:chartData,options:chartOpts});

  const rows=bs.map(b=>{
    if(b.is_tail){
      return`<tr class="tail-row">
        <td style="text-align:left;color:#7c3aed;font-size:13px">${b.label}</td>
        <td style="color:#94a3b8">—</td>
        <td style="color:#7c3aed;font-weight:700">${b.opt_prob!==null?b.opt_prob.toFixed(1)+'%':'—'}</td>
        <td style="color:#94a3b8" colspan="3">probability mass outside bracket coverage</td>
      </tr>`;
    }
    const ec=b.edge>3?'under':b.edge<-3?'over':'fair';
    const sig=b.edge>3?'UNDER':b.edge<-3?'OVER':'FAIR';
    const sign=b.edge>=0?'+':'';
    const bandStr=b.half_spread?`<span class="spread-band"> ± ${b.half_spread.toFixed(1)}%</span>`:'';
    const srcBadge=b.poly_source==='clob'?'<span class="clob-badge">CLOB</span>':'<span class="gamma-badge">MARK</span>';
    return`<tr>
      <td style="text-align:left;font-weight:700;color:#374151;font-size:15px">${b.label}</td>
      <td style="color:#2563eb;font-weight:700">${b.poly_prob!==null?b.poly_prob.toFixed(1)+'%':'—'}${srcBadge}</td>
      <td style="color:#7c3aed;font-weight:700">${b.opt_prob!==null?b.opt_prob.toFixed(1)+'%':'—'}</td>
      <td class="${ec}">${b.edge!==null?sign+b.edge.toFixed(1)+'%'+bandStr:'—'}</td>
      <td><span class="pill pill-${ec}">${sig}</span></td>
      <td style="color:#94a3b8">$${Number(b.volume).toLocaleString('en-US',{maximumFractionDigits:0})}</td>
    </tr>`;
  }).join('');
  document.getElementById('bracket-tbody').innerHTML=rows;
}

function renderSmiles(surface,targetExpiry){
  const exps=[...new Set(surface.map(o=>o.expiry))].sort();
  const exp=(targetExpiry&&exps.includes(targetExpiry))?targetExpiry
            :exps.find(e=>e>=new Date().toISOString().slice(0,10))||exps[0];
  const sel=document.getElementById('expiry-sel');
  while(sel.options.length>1)sel.remove(1);
  exps.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;sel.appendChild(o);});
  const slice=surface.filter(o=>o.expiry===exp);
  document.getElementById('smile-expiry').textContent=exp;
  document.getElementById('iv-expiry').textContent=exp;
  const labs=slice.map(o=>fmt$(o.strike));
  const probs=slice.map(o=>+(o.prob_itm*100).toFixed(1));
  const ivs=slice.map(o=>+o.iv.toFixed(1));
  const mkDs=(data,color)=>({data,borderColor:color,
    backgroundColor:color.replace('rgb(','rgba(').replace(')',',0.07)'),
    tension:0.3,pointRadius:3,pointHoverRadius:7,fill:true,borderWidth:2.5});
  const shTip={backgroundColor:'#1e293b',titleColor:'#f8fafc',bodyColor:'#cbd5e1',borderColor:'#334155',borderWidth:1,titleFont:{size:14,weight:'600'},bodyFont:{size:13},padding:12};
  if(smileChart){smileChart.data.labels=labs;smileChart.data.datasets[0].data=probs;smileChart.update();}
  else smileChart=new Chart(document.getElementById('smileChart').getContext('2d'),
    {type:'line',data:{labels:labs,datasets:[mkDs(probs,'rgb(124,58,237)')]},
     options:{...baseOpts('P(ITM) %','Strike'),plugins:{legend:{display:false},tooltip:shTip}}});
  if(ivChart){ivChart.data.labels=labs;ivChart.data.datasets[0].data=ivs;ivChart.update();}
  else ivChart=new Chart(document.getElementById('ivChart').getContext('2d'),
    {type:'line',data:{labels:labs,datasets:[mkDs(ivs,'rgb(5,150,105)')]},
     options:{...baseOpts('Implied Vol %','Strike'),plugins:{legend:{display:false},tooltip:shTip}}});
}

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

function render(d){
  document.getElementById('loading').style.display='none';
  document.getElementById('spot').textContent='$'+d.spot.toLocaleString('en-US',{maximumFractionDigits:0});
  document.getElementById('deribit-spot').textContent=d.deribit_spot?'$'+d.deribit_spot.toLocaleString('en-US',{maximumFractionDigits:0}):'—';
  const basisVal=d.basis;
  document.getElementById('basis').textContent=basisVal!==null?((basisVal>=0?'+':'')+basisVal.toLocaleString('en-US',{maximumFractionDigits:0})):'—';
  document.getElementById('nd').textContent=d.n_deribit;
  document.getElementById('ne').textContent=d.n_events;
  document.getElementById('upd').textContent=d.updated+' ('+d.elapsed+'s)';
  document.getElementById('atm-iv').textContent=d.atm_iv?d.atm_iv+'%':'—';
  document.getElementById('realized-vol').textContent=d.realized_vol?d.realized_vol+'%':'—';
  document.getElementById('rfr').textContent=d.risk_free||'4.3';

  const tabsEl=document.getElementById('event-tabs');
  const noEvPanel=document.getElementById('no-events-panel');
  const mainWrap=document.getElementById('main-chart-wrap');
  tabsEl.innerHTML='';

  if(!d.events.length){
    noEvPanel.style.display='block';mainWrap.style.display='none';
    document.getElementById('bracket-tbody').innerHTML='';
  } else {
    noEvPanel.style.display='none';mainWrap.style.display='block';
    d.events.forEach((ev,i)=>{
      const btn=document.createElement('button');
      btn.className='etab'+(i===activeEventIdx?' active':'');
      btn.textContent=ev.end_label;
      btn.onclick=()=>{
        activeEventIdx=i;
        document.querySelectorAll('.etab').forEach((b,j)=>b.className='etab'+(j===i?' active':''));
        renderEvent(d.events,i);
        renderSmiles(d.vol_surface,d.events[i].deribit_expiry.split(' ')[0]);
      };
      tabsEl.appendChild(btn);
    });
    if(activeEventIdx>=d.events.length)activeEventIdx=0;
    renderEvent(d.events,activeEventIdx);
  }
  const smileExpiry=d.events.length?d.events[activeEventIdx]?.deribit_expiry.split(' ')[0]:null;
  renderSmiles(d.vol_surface,smileExpiry);
  renderSurface(d.vol_surface,'all');
  document.getElementById('expiry-sel').addEventListener('change',e=>{renderSurface(d.vol_surface,e.target.value);});
}

async function load(){
  try{
    const res=await fetch('/api/data');
    const j=await res.json();
    if(j.error){document.getElementById('upd').textContent='Error: '+j.error;return;}
    allData=j.data;render(allData);
  }catch(e){console.error(e);}
}
function tick(){countdown--;document.getElementById('ct').textContent=countdown;if(countdown<=0){countdown=REFRESH;load();}}
load();setInterval(tick,1000);
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
        self.send_response(code); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    print("Starting data fetch...")
    threading.Thread(target=_loop, daemon=True).start()
    host = "0.0.0.0"
    url  = f"http://localhost:{PORT}"
    print(f"Dashboard -> {url}   (Ctrl-C to stop)")
    if not os.environ.get("RENDER"):
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    try: HTTPServer((host, PORT), Handler).serve_forever()
    except KeyboardInterrupt: print("\nStopped.")
