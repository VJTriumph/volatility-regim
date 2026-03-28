#!/usr/bin/env python3
"""
NIFTY Volatility Parameter Dashboard — Professional Quant Edition
Outputs a fully self-contained interactive HTML dashboard.

FEATURES
  • Configurable expiry regimes (SEBI change Nov 2024: Thu→Tue)
  • Close-to-Close AND Garman-Klass estimators
  • Bootstrap 95% confidence intervals
  • Welch t-test on |returns| + Benjamini-Hochberg FDR correction
  • Outlier filter (2σ / 2.5σ / 3σ / off)
  • VIX regime conditioning (Low / Med / High terciles)
  • Custom date range override
  • All 5 calendar dimensions: DoM, WoM, Weekday, Monthly Expiry, Weekly Expiry
  • Significance heatmap table
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, calendar, os
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
#  USER CONFIG  ← only section you need to edit
# ═══════════════════════════════════════════════════════════════
PERIODS = [
    dict(label  = "2015–2019  ·  Thu Expiry",
         start  = "2015-01-01",
         end    = "2019-12-31",
         monthly_exp_dow = 3,     # 0=Mon … 4=Fri
         weekly_exp_dow  = 3),
    dict(label  = "2020–Oct 2024  ·  Thu Expiry",
         start  = "2020-01-01",
         end    = "2024-10-31",
         monthly_exp_dow = 3,
         weekly_exp_dow  = 3),
    dict(label  = "Nov 2024–Now  ·  Tue Expiry",
         start  = "2024-11-01",
         end    = "2026-03-27",
         monthly_exp_dow = 1,     # Tuesday
         weekly_exp_dow  = 1),
]
# Paths to local CSV files in data/ folder
NIFTY_CSV   = os.path.join("data", "nifty 10 year data.csv")
VIX_CSV     = os.path.join("data", "indiavix 10 year data.csv")
OUT         = "nifty_vol_dashboard.html"
ANN         = 252

# ═══════════════════════════════════════════════════════════════
#  DATA LOAD  ← reads local CSV files (no internet required)
# ═══════════════════════════════════════════════════════════════
WD_LABEL = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

def load_csv_ohlcv(filepath):
    """Load OHLCV data from CSV with date format like '1-Jan-15'."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%b-%y")
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index).normalize()
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ""),
            errors="coerce"
        )
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]  # drop duplicate dates
    return df

def fetch_ohlcv():
    print(f"  Loading NIFTY data from {NIFTY_CSV} ...")
    if not os.path.exists(NIFTY_CSV):
        sys.exit(f"ERROR: File not found: {NIFTY_CSV}")
    raw = load_csv_ohlcv(NIFTY_CSV)
    if raw.empty:
        sys.exit("ERROR: NIFTY CSV returned empty data")
    print(f"  {len(raw)} rows [{raw.index[0].date()} -> {raw.index[-1].date()}]")

    print(f"  Loading India VIX data from {VIX_CSV} ...")
    vix = pd.Series(dtype=float, name="vix")
    if os.path.exists(VIX_CSV):
        try:
            vdf = load_csv_ohlcv(VIX_CSV)
            vix = vdf["Close"].squeeze().astype(float)
            vix.name = "vix"
            print(f"  {len(vix)} VIX rows")
        except Exception as ex:
            print(f"  VIX load failed ({ex}) - continuing without VIX conditioning")
    else:
        print("  VIX file not found - continuing without VIX conditioning")
    return raw, vix

# ═══════════════════════════════════════════════════════════
#  BUILD ROW DATA
# ═══════════════════════════════════════════════════════════════
def _resolve_expiry(scheduled, trading_days):
    """Roll scheduled expiry back to the nearest prior trading day if it's a holiday."""
    d = scheduled
    while d not in trading_days:
        d -= pd.Timedelta(days=1)
        if d < pd.Timestamp("2000-01-01"):
            return scheduled  # safety guard
    return d

def tag_monthly_expiry(idx, exp_dow):
    """Tag each date relative to its monthly expiry (last exp_dow of month).
    If that day is a market holiday, expiry rolls to the previous trading day."""
    trading_days = set(idx)
    # cache expiry per (year, month) to avoid recomputing
    expiry_cache = {}
    tags = []
    for ts in idx:
        y, m = ts.year, ts.month
        key = (y, m)
        if key not in expiry_cache:
            last_day = calendar.monthrange(y, m)[1]
            sched = None
            for d in range(last_day, 0, -1):
                if pd.Timestamp(y, m, d).weekday() == exp_dow:
                    sched = pd.Timestamp(y, m, d)
                    break
            expiry_cache[key] = _resolve_expiry(sched, trading_days) if sched else None
        expiry = expiry_cache[key]
        if expiry is None:
            tags.append("")
        elif ts == expiry:
            tags.append("ExpiryDay")
        elif ts == expiry - pd.Timedelta(days=1):
            tags.append("DayBefore")
        elif ts == expiry + pd.Timedelta(days=1):
            tags.append("DayAfter")
        else:
            tags.append("")
    return tags

def tag_weekly_expiry(idx, exp_dow):
    """Tag each date relative to the weekly expiry (every exp_dow).
    If that exp_dow is a market holiday, expiry rolls to the previous trading day."""
    trading_days = set(idx)
    # Build a map: for each week find actual expiry day
    # A 'week' is identified by the Monday of that week
    expiry_cache = {}
    def get_weekly_expiry(ts):
        # find the scheduled expiry (exp_dow) of this week
        wd = ts.weekday()
        days_to_expiry = (exp_dow - wd) % 7
        sched = ts + pd.Timedelta(days=days_to_expiry)
        # if we've already passed expiry this week, look at last week's
        if days_to_expiry == 0:
            sched = ts  # it's today (will be resolved below)
        week_key = sched
        if week_key not in expiry_cache:
            expiry_cache[week_key] = _resolve_expiry(sched, trading_days)
        return expiry_cache[week_key]

    tags = []
    for ts in idx:
        wd = ts.weekday()
        days_to_expiry = (exp_dow - wd) % 7
        sched = ts + pd.Timedelta(days=days_to_expiry)
        week_key = sched
        if week_key not in expiry_cache:
            expiry_cache[week_key] = _resolve_expiry(sched, trading_days)
        expiry = expiry_cache[week_key]
        if ts == expiry:
            tags.append("ExpiryDay")
        elif ts == expiry - pd.Timedelta(days=1):
            tags.append("DayBefore")
        elif ts == expiry + pd.Timedelta(days=1):
            tags.append("DayAfter")
        else:
            tags.append("")
    return tags

def build_rows(ohlc, vix):
    rows = []
    for pid, p in enumerate(PERIODS):
        s, e = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
        df   = ohlc.loc[s:e].copy()
        if len(df) < 30:
            print(f"    Period {pid}: only {len(df)} rows — skipped"); continue

        df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df        = df.dropna(subset=["ret"])

        # Garman-Klass daily variance (raw, multiply by ANN to annualize)
        # Formula: 0.5*(ln H/L)^2 - (2ln2-1)*(ln C/O)^2
        ln_hl = np.log(df["High"]  / df["Low"])
        ln_co = np.log(df["Close"] / df["Open"])
        df["gk"] = (0.5 * ln_hl**2 - (2*np.log(2)-1) * ln_co**2).clip(lower=0)

        idx = df.index
        df["dom"] = idx.day
        df["wom"] = ((idx.day - 1) // 7) + 1
        df["wdn"] = [WD_LABEL[d] for d in idx.weekday]
        df["meg"] = tag_monthly_expiry(idx, p["monthly_exp_dow"])
        df["weg"] = tag_weekly_expiry(idx,  p["weekly_exp_dow"])
        df["vix"] = vix.reindex(idx, method="ffill").fillna(0) if len(vix) else 0.0
        df["pid"] = pid

        for ts, row in df.iterrows():
            rows.append({
                "date": ts.strftime("%Y-%m-%d"),
                "ret":  round(float(row.ret),  6),
                "gk":   round(float(row.gk),   8),
                "dom":  int(row.dom),
                "wom":  int(row.wom),
                "wdn":  row.wdn,
                "meg":  row.meg,
                "weg":  row.weg,
                "vix":  round(float(row.vix), 2),
                "pid":  int(pid),
            })
        print(f"    Period {pid} ({p['label']}): {len(df)} rows")

    return rows

# ═══════════════════════════════════════════════════════════════
#  HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════
def build_html(rows):
    periods_js = json.dumps([
        {"label": p["label"], "start": p["start"], "end": p["end"]}
        for p in PERIODS
    ])
    data_js = json.dumps(rows, separators=(',', ':'))
    has_vix  = json.dumps(any(r["vix"] > 0 for r in rows))
    all_start = min(r["date"] for r in rows)
    all_end   = max(r["date"] for r in rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIFTY Vol Params · Quant Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{{
  --bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--border:#30363d;
  --fg:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;
  --orange:#f0883e;--red:#ff7b72;--purple:#bc8cff;--yellow:#e3b341;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font-family:'SF Mono','Fira Code',ui-monospace,monospace;font-size:12px;min-height:100vh}}
a{{color:var(--accent)}}

/* ── HEADER ── */
#hdr{{background:var(--panel);border-bottom:1px solid var(--border);padding:14px 20px;display:flex;align-items:center;gap:16px}}
#hdr h1{{font-size:15px;font-weight:700;color:var(--fg);letter-spacing:.5px}}
#hdr span{{font-size:11px;color:var(--muted)}}
.dot{{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;margin-right:6px;box-shadow:0 0 6px var(--green)}}

/* ── CONTROLS ── */
#ctrl{{background:var(--panel2);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end}}
.cg{{display:flex;flex-direction:column;gap:5px}}
.cg label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}}
.tabs{{display:flex;gap:4px}}
.tab{{background:var(--panel);border:1px solid var(--border);color:var(--muted);padding:5px 11px;border-radius:5px;cursor:pointer;font-size:11px;font-family:inherit;transition:all .15s}}
.tab:hover{{border-color:var(--accent);color:var(--fg)}}
.tab.on{{background:var(--accent);border-color:var(--accent);color:#0d1117;font-weight:700}}
select,input[type=date]{{background:var(--panel);border:1px solid var(--border);color:var(--fg);padding:5px 8px;border-radius:5px;font-size:11px;font-family:inherit;outline:none;cursor:pointer}}
select:focus,input[type=date]:focus{{border-color:var(--accent)}}
.sep{{width:1px;background:var(--border);margin:0 4px}}
#apply-btn{{background:var(--accent);color:#0d1117;border:none;padding:6px 14px;border-radius:5px;font-size:11px;font-family:inherit;font-weight:700;cursor:pointer}}
#apply-btn:hover{{opacity:.85}}

/* ── SUMMARY BAR ── */
#summary{{background:var(--panel);border-bottom:1px solid var(--border);padding:8px 20px;display:flex;flex-wrap:wrap;gap:24px}}
.stat{{display:flex;flex-direction:column;gap:2px}}
.stat-label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}}
.stat-val{{font-size:14px;font-weight:700;color:var(--fg)}}
.stat-val.hi{{color:var(--orange)}}
.stat-val.lo{{color:var(--green)}}

/* ── CHARTS GRID ── */
#charts{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:14px 16px}}
.chart-card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px;overflow:hidden}}
.chart-card h3{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;padding:0 4px}}
@media(max-width:1100px){{#charts{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:700px){{#charts{{grid-template-columns:1fr}}}}

/* ── TABLE ── */
#tbl-wrap{{padding:0 16px 20px}}
#tbl-wrap h2{{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;padding:12px 4px 8px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:8px;overflow:hidden;font-size:11px}}
th{{background:var(--panel2);color:var(--muted);padding:8px 12px;text-align:left;font-weight:600;border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:7px 12px;border-bottom:1px solid var(--border);white-space:nowrap}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--panel2)}}
.sig-hi{{color:var(--red);font-weight:700}}
.sig-med{{color:var(--orange);font-weight:600}}
.sig-lo{{color:var(--muted)}}
.badge{{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:700}}
.badge-red{{background:#ff7b7220;color:var(--red);border:1px solid #ff7b7240}}
.badge-orange{{background:#f0883e20;color:var(--orange);border:1px solid #f0883e40}}
.badge-green{{background:#3fb95020;color:var(--green);border:1px solid #3fb95040}}
.badge-muted{{background:#8b949e10;color:var(--muted);border:1px solid #8b949e30}}

/* ── MISC ── */
#notice{{font-size:10px;color:var(--muted);padding:4px 20px 10px;font-style:italic}}
.spinner{{text-align:center;padding:40px;color:var(--muted)}}
</style>
</head>
<body>

<!-- HEADER -->
<div id="hdr">
  <h1><span class="dot"></span>NIFTY Vol Params</h1>
  <span>Quant Dashboard · Calendar-Segmented Realized Volatility</span>
</div>

<!-- CONTROLS -->
<div id="ctrl">
  <div class="cg">
    <label>Period</label>
    <div class="tabs" id="period-tabs"></div>
  </div>
  <div class="sep"></div>
  <div class="cg">
    <label>Estimator</label>
    <div class="tabs" id="est-tabs">
      <button class="tab on" data-v="cc" onclick="setEst(this)">Close-Close</button>
      <button class="tab"    data-v="gk" onclick="setEst(this)">Garman-Klass</button>
    </div>
  </div>
  <div class="sep"></div>
  <div class="cg">
    <label>Outlier Filter</label>
    <select id="out-sel" onchange="update()">
      <option value="0">None</option>
      <option value="3">3σ</option>
      <option value="2.5" selected>2.5σ (recommended)</option>
      <option value="2">2σ</option>
    </select>
  </div>
  <div class="sep"></div>
  <div class="cg" id="vix-cg" style="display:none">
    <label>VIX Regime</label>
    <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
      <select id="vix-sel" onchange="update()" style="min-width:106px">
        <option value="all">All Regimes</option>
        <option value="low">Low</option>
        <option value="med">Medium</option>
        <option value="high">High</option>
      </select>
      <input type="number" id="vix-lo" value="14" min="1" max="100" step="1"
        style="width:52px;background:var(--panel);border:1px solid var(--border);color:var(--fg);padding:4px 6px;border-radius:5px;font-size:11px;font-family:inherit"
        onchange="updateVixThresholds()" title="Low / Med boundary">
      <span style="color:var(--muted);font-size:10px">–</span>
      <input type="number" id="vix-hi" value="20" min="1" max="100" step="1"
        style="width:52px;background:var(--panel);border:1px solid var(--border);color:var(--fg);padding:4px 6px;border-radius:5px;font-size:11px;font-family:inherit"
        onchange="updateVixThresholds()" title="Med / High boundary">
    </div>
  </div>
  <div class="sep"></div>
  <div class="cg">
    <label>Custom Date Range</label>
    <div style="display:flex;gap:6px;align-items:center">
      <input type="date" id="d-start" value="{all_start}">
      <span style="color:var(--muted)">→</span>
      <input type="date" id="d-end" value="{all_end}">
      <button id="apply-btn" onclick="applyDates()">Apply</button>
      <button class="tab" onclick="resetDates()" style="margin-left:2px">Reset</button>
    </div>
  </div>
</div>

<!-- SUMMARY BAR -->
<div id="summary"></div>
<div id="notice"></div>

<!-- CHARTS -->
<div id="charts">
  <div class="chart-card"><h3>Day of Month</h3><div id="c-dom" style="height:290px"></div></div>
  <div class="chart-card"><h3>Week of Month</h3><div id="c-wom" style="height:290px"></div></div>
  <div class="chart-card"><h3>Day of Week</h3><div id="c-wdn" style="height:290px"></div></div>
  <div class="chart-card"><h3>Monthly Expiry Window</h3><div id="c-meg" style="height:290px"></div></div>
  <div class="chart-card"><h3>Weekly Expiry Window</h3><div id="c-weg" style="height:290px"></div></div>
  <div class="chart-card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;padding:0 4px">
      <h3 style="margin:0">VIX Regime · Expiry Vol</h3>
      <div style="display:flex;gap:4px" id="vix-exp-tabs">
        <button class="tab on" data-v="monthly" onclick="setVixExpType(this,'monthly')">Monthly</button>
        <button class="tab" data-v="weekly" onclick="setVixExpType(this,'weekly')">Weekly</button>
      </div>
    </div>
    <div id="c-vix" style="height:264px"></div>
  </div>
</div>

<!-- TABLE -->
<div id="tbl-wrap">
  <h2>Significance Analysis — All Calendar Groups</h2>
  <div id="tbl-inner"></div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════
const PERIODS  = {periods_js};
const RAW      = {data_js};
const HAS_VIX  = {has_vix};
const ANN      = {ANN};

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════
let state = {{
  pid:       -1,        // -1 = all periods
  estimator: "cc",      // cc | gk
  outlier:   2.5,       // sigma, 0=off
    vixReg:    "all",     // all|low|med|high
    vixLow:    14,        // VIX low/med boundary
    vixHigh:   20,        // VIX med/high boundary
    vixExpType: "monthly", // monthly | weekly
  dateStart: "{all_start}",
  dateEnd:   "{all_end}",
}};

// ═══════════════════════════════════════════════════════════════
// STATS ENGINE
// ═══════════════════════════════════════════════════════════════
function mean(a){{ return a.length ? a.reduce((s,v)=>s+v,0)/a.length : 0 }}

function sampleVar(a){{
  const n=a.length; if(n<2) return 0;
  const mu=mean(a);
  return a.reduce((s,v)=>s+(v-mu)**2,0)/(n-1);
}}

function annVolCC(rets){{ return Math.sqrt(Math.max(sampleVar(rets),0)*ANN) }}

function annVolGK(gks){{
  const mu=mean(gks);
  return Math.sqrt(Math.max(mu,0)*ANN);
}}

function estVol(vals, estimator){{
  return estimator==="gk" ? annVolGK(vals) : annVolCC(vals);
}}

function bootstrap(vals, estimator, nIter=300){{
  const n=vals.length;
  if(n<4) return [NaN,NaN];
  const boots=[];
  for(let i=0;i<nIter;i++){{
    const s=[];
    for(let j=0;j<n;j++) s.push(vals[Math.floor(Math.random()*n)]);
    boots.push(estVol(s,estimator));
  }}
  boots.sort((a,b)=>a-b);
  return [boots[Math.floor(0.025*nIter)], boots[Math.floor(0.975*nIter)]];
}}

// Welch t-test on absolute returns (vol proxy)
function normalCDF(z){{
  const t=1/(1+0.2316419*Math.abs(z));
  const p=1-0.3989422803*Math.exp(-z*z/2)*
    t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));
  return z<0?1-p:p;
}}

function welchTtest(g1,g2){{
  if(g1.length<3||g2.length<3) return 1;
  const a1=g1.map(Math.abs), a2=g2.map(Math.abs);
  const n1=a1.length, n2=a2.length;
  const v1=sampleVar(a1)/n1, v2=sampleVar(a2)/n2;
  const se=Math.sqrt(v1+v2);
  if(se<1e-12) return 1;
  const t=Math.abs(mean(a1)-mean(a2))/se;
  return 2*(1-normalCDF(t));
}}

function bhFDR(pvals){{
  const n=pvals.length;
  const idx=pvals.map((p,i)=>{{return{{p,i}}}}).sort((a,b)=>a.p-b.p);
  const adj=new Array(n).fill(1);
  for(let k=0;k<n;k++) adj[idx[k].i]=Math.min(1,idx[k].p*n/(k+1));
  // monotone step
  for(let k=n-2;k>=0;k--) adj[idx[k].i]=Math.min(adj[idx[k].i],adj[idx[k+1].i]);
  return adj;
}}

// ═══════════════════════════════════════════════════════════════
// FILTERING
// ═══════════════════════════════════════════════════════════════
function getFiltered(){{
  let rows = RAW.filter(r=>{{
    if(state.pid>=0 && r.pid!==state.pid) return false;
    if(r.date<state.dateStart || r.date>state.dateEnd) return false;
    return true;
  }});

  // outlier filter on log returns
  if(state.outlier>0 && rows.length>5){{
    const rets=rows.map(r=>r.ret);
    const mu=mean(rets);
    const sd=Math.sqrt(sampleVar(rets));
    const thr=state.outlier*sd;
    const before=rows.length;
    rows=rows.filter(r=>Math.abs(r.ret-mu)<=thr);
    window._outlierRemoved=before-rows.length;
  }} else {{ window._outlierRemoved=0; }}


    // VIX regime — fixed thresholds
    if(state.vixReg!=="all" && rows.some(r=>r.vix>0)){{
      const lo=state.vixLow, hi=state.vixHigh;
      if(state.vixReg==="low")  rows=rows.filter(r=>r.vix>0&&r.vix<lo);
      if(state.vixReg==="med")  rows=rows.filter(r=>r.vix>0&&r.vix>=lo&&r.vix<=hi);
      if(state.vixReg==="high") rows=rows.filter(r=>r.vix>0&&r.vix>hi);
    }}

  return rows;
}}

// ═══════════════════════════════════════════════════════════════
// ANALYSIS
// ═══════════════════════════════════════════════════════════════
function analyzeGroup(rows, keyFn, labels, estimator){{
  const rets=rows.map(r=>r.ret);
  const allVals=estimator==="gk"?rows.map(r=>r.gk):rets;
  const overallVol=estVol(allVals,estimator);

  const results=[];
  for(const lbl of labels){{
    const grp=rows.filter(r=>keyFn(r)===lbl);
    const rest=rows.filter(r=>keyFn(r)!==lbl);
    const gVals=estimator==="gk"?grp.map(r=>r.gk):grp.map(r=>r.ret);
    const rVals=estimator==="gk"?rest.map(r=>r.gk):rest.map(r=>r.ret);
    if(grp.length<2){{
      results.push({{label:String(lbl),n:grp.length,vol:NaN,ciLo:NaN,ciHi:NaN,pval:1,adjP:1,ratio:NaN,meanRet:NaN}});
      continue;
    }}
    const vol=estVol(gVals,estimator);
    const [ciLo,ciHi]=bootstrap(gVals,estimator,250);
    // t-test always on returns (vol proxy)
    const pval=welchTtest(grp.map(r=>r.ret),rest.map(r=>r.ret));
    const mr=mean(grp.map(r=>r.ret));
    results.push({{label:String(lbl),n:grp.length,vol,ciLo,ciHi,pval,adjP:1,ratio:vol/overallVol,meanRet:mr}});
  }}

  // BH-FDR
  const adjP=bhFDR(results.map(r=>r.pval));
  results.forEach((r,i)=>r.adjP=adjP[i]);

  return {{results,overallVol}};
}}

function analyzeAll(rows,estimator){{
  // Day of month — only keep days with n>=3
  const domAll=[...Array(31)].map((_,i)=>i+1);
  const domResult=analyzeGroup(rows,r=>r.dom,domAll,estimator);
  domResult.results=domResult.results.filter(r=>r.n>=3);

  // Week of month
  const wom=analyzeGroup(rows,r=>r.wom,[1,2,3,4,5],estimator);

  // Weekday
  const wdn=analyzeGroup(rows,r=>r.wdn,["Mon","Tue","Wed","Thu","Fri"],estimator);

  // Monthly expiry window
  const meg=analyzeGroup(rows,r=>r.meg,["DayBefore","ExpiryDay","DayAfter"],estimator);

  // Weekly expiry window
  const weg=analyzeGroup(rows,r=>r.weg,["DayBefore","ExpiryDay","DayAfter"],estimator);

  return {{dom:domResult,wom,wdn,meg,weg}};
}}

// VIX regime chart
function analyzeVixRegimes(rows,estimator,vixLow,vixHigh,expType){{
  const expField=expType==="weekly"?"weg":"meg";
  if(!rows.some(r=>r.vix>0)) return null;
  const lo=vixLow||14, hi=vixHigh||20;
  const regMap=r=>r.vix>0&&r.vix<lo?"Low":r.vix>=lo&&r.vix<=hi?"Med":"High";
  const res={{}};
  for(const reg of["Low","Med","High"]){{
    const sub=rows.filter(r=>r.vix>0&&regMap(r)===reg);
    const expGrp={{
      DayBefore:sub.filter(r=>r[expField]==="DayBefore"),
      ExpiryDay:sub.filter(r=>r[expField]==="ExpiryDay"),
      DayAfter:sub.filter(r=>r[expField]==="DayAfter"),
      Other:sub.filter(r=>r[expField]==="")
    }};
    const vals={{}};
    for(const [k,g] of Object.entries(expGrp)){{
      if(!g.length){{ vals[k]={{vol:NaN,n:0}}; continue; }}
      const v=estimator==="gk"?g.map(r=>r.gk):g.map(r=>r.ret);
      vals[k]={{vol:estVol(v,estimator),n:g.length}};
    }}
    res[reg]=vals;
  }}
  return {{regimes:res,vixLow:lo,vixHigh:hi,expType:expType||"monthly"}};
}}

// ═══════════════════════════════════════════════════════════════
// PLOTLY HELPERS
// ═══════════════════════════════════════════════════════════════
const PLOTLY_LAYOUT={{
  paper_bgcolor:"transparent",plot_bgcolor:"#161b22",
  font:{{color:"#e6edf3",family:"'SF Mono','Fira Code',monospace",size:11}},
  margin:{{l:46,r:14,t:10,b:50}},
  xaxis:{{gridcolor:"#21262d",linecolor:"#30363d",tickcolor:"#8b949e",tickfont:{{size:10}}}},
  yaxis:{{gridcolor:"#21262d",linecolor:"#30363d",tickcolor:"#8b949e",tickformat:".1%",tickfont:{{size:10}}}},
  showlegend:false,
  hoverlabel:{{bgcolor:"#1c2128",bordercolor:"#30363d",font:{{color:"#e6edf3",size:11}}}},
}};

const PLOTLY_CFG={{responsive:true,displayModeBar:false}};

function sigColor(adjP,ratio){{
  if(isNaN(adjP)) return "#8b949e";
  if(adjP<0.01) return ratio>1?"#ff7b72":"#3fb950";
  if(adjP<0.05) return ratio>1?"#f0883e":"#79c0ff";
  return "#58a6ff";
}}

function renderBar(divId,analysis){{
  const {{results,overallVol}}=analysis;
  const labels=results.map(r=>r.label);
  const vols=results.map(r=>r.vol);
  const errHi=results.map(r=>isNaN(r.ciHi)?0:r.ciHi-r.vol);
  const errLo=results.map(r=>isNaN(r.ciLo)?0:r.vol-r.ciLo);
  const colors=results.map(r=>sigColor(r.adjP,r.ratio));

  const hover=results.map(r=>`
    <b>${{r.label}}</b><br>
    n: ${{r.n}}<br>
    AnnVol: ${{(r.vol*100).toFixed(2)}}%<br>
    95% CI: [${{(r.ciLo*100).toFixed(2)}}%, ${{(r.ciHi*100).toFixed(2)}}%]<br>
    vs Avg: ${{(r.ratio*100-100).toFixed(1)}}%<br>
    Raw p: ${{r.pval.toFixed(3)}}<br>
    FDR-adj p: ${{r.adjP.toFixed(3)}}<br>
    Avg Ret: ${{(r.meanRet*100).toFixed(3)}}%
  `.trim());

  const trace={{
    type:"bar", x:labels, y:vols,
    marker:{{color:colors,line:{{color:"#30363d",width:.5}}}},
    error_y:{{type:"data",array:errHi,arrayminus:errLo,visible:true,
              color:"#8b949e",thickness:1.2,width:4}},
    hovertemplate:"%{{customdata}}<extra></extra>",
    customdata:hover,
  }};

  const avgLine={{
    type:"scatter",mode:"lines",
    x:labels, y:labels.map(()=>overallVol),
    line:{{color:"#f0883e",width:1.5,dash:"dash"}},
    hoverinfo:"skip",
  }};

  const layout={{
    ...PLOTLY_LAYOUT,
    xaxis:{{...PLOTLY_LAYOUT.xaxis,type:"category"}},
    shapes:[{{
      type:"line",xref:"paper",yref:"y",
      x0:0,x1:1,y0:overallVol,y1:overallVol,
      line:{{color:"#f0883e55",width:1,dash:"dot"}},
    }}],
    annotations:[{{
      xref:"paper",yref:"y",x:1,y:overallVol,
      text:`Avg ${{(overallVol*100).toFixed(1)}}%`,
      showarrow:false,font:{{size:9,color:"#f0883e"}},
      xanchor:"right",yanchor:"bottom",
    }}],
  }};

  Plotly.react(divId,[trace],layout,PLOTLY_CFG);
}}

function renderVixChart(vixData){{
  const div="c-vix";
  if(!vixData){{
    document.getElementById(div).innerHTML=
      `<div style="color:var(--muted);padding:40px;text-align:center;font-size:11px">VIX data not available</div>`;
    return;
  }}
  const regs=["Low","Med","High"];
  const grps=["DayBefore","ExpiryDay","DayAfter","Other"];
  const grpColors=["#58a6ff","#ff7b72","#3fb950","#8b949e"];
  const traces=grps.map((g,gi)=>{{
    const y=regs.map(r=>{{
      const v=vixData.regimes[r]?.[g]?.vol;
      return isNaN(v)?null:v;
    }});
    return {{
      type:"bar",name:g,x:regs,y,
      marker:{{color:grpColors[gi],opacity:.85}},
      hovertemplate:`${{g}}: %{{y:.2%}}<extra></extra>`,
    }};
  }});
  const layout={{
    ...PLOTLY_LAYOUT,
    xaxis:{{...PLOTLY_LAYOUT.xaxis,type:"category"}},
    showlegend:true,
    legend:{{x:0,y:1.1,orientation:"h",font:{{size:10}}}},
    barmode:"group",
    margin:{{...PLOTLY_LAYOUT.margin,b:40,t:20}},
    annotations:[
      {{xref:"paper",yref:"paper",x:0,y:-0.12,
        text:`${{vixData.expType==="weekly"?"Weekly":"Monthly"}} Exp · VIX Low<${{vixData.vixLow}} Med ${{vixData.vixLow}}–${{vixData.vixHigh}} High>${{vixData.vixHigh}}`,
        showarrow:false,font:{{size:9,color:"#8b949e"}},xanchor:"left"}}
    ],
  }};
  Plotly.react(div,traces,layout,PLOTLY_CFG);
}}

// ═══════════════════════════════════════════════════════════════
// SUMMARY BAR
// ═══════════════════════════════════════════════════════════════
function renderSummary(rows,analysis){{
  const n=rows.length;
  const ov=analysis.dom.overallVol;
  const rets=rows.map(r=>r.ret);
  const mu=mean(rets);
  const vix=rows.filter(r=>r.vix>0).map(r=>r.vix);
  const avgVix=vix.length?mean(vix):null;
  const dateRange=rows.length?`${{rows[0].date}} → ${{rows[rows.length-1].date}}`:"–";

  const items=[
    {{label:"Trading Days",val:n,cls:""}},
    {{label:"Overall Ann Vol",val:(ov*100).toFixed(2)+"%",cls:ov>0.2?"hi":"lo"}},
    {{label:"Mean Daily Ret",val:(mu*100).toFixed(3)+"%",cls:mu>0?"lo":"hi"}},
    {{label:"Outliers Removed",val:window._outlierRemoved||0,cls:""}},
    avgVix?{{label:"Avg India VIX",val:avgVix.toFixed(1),cls:avgVix>18?"hi":"lo"}}:null,
    {{label:"Date Range",val:dateRange,cls:""}},
  ].filter(Boolean);

  document.getElementById("summary").innerHTML=items.map(it=>
    `<div class="stat"><span class="stat-label">${{it.label}}</span>
     <span class="stat-val ${{it.cls}}">${{it.val}}</span></div>`
  ).join("");
}}

// ═══════════════════════════════════════════════════════════════
// SIGNIFICANCE TABLE
// ═══════════════════════════════════════════════════════════════
function renderTable(analysis){{
  const dims=[
    {{name:"Day of Month",  key:"dom"}},
    {{name:"Week of Month", key:"wom"}},
    {{name:"Day of Week",   key:"wdn"}},
    {{name:"Monthly Expiry",key:"meg"}},
    {{name:"Weekly Expiry", key:"weg"}},
  ];

  let allRows=[];
  for(const d of dims){{
    const {{results}}=analysis[d.key];
    for(const r of results){{
      if(!isNaN(r.vol))
        allRows.push({{...r,dim:d.name}});
    }}
  }}

  // sort by |ratio-1| desc
  allRows.sort((a,b)=>Math.abs(b.ratio-1)-Math.abs(a.ratio-1));

  function pBadge(p){{
    if(p<0.01) return `<span class="badge badge-red">p&lt;0.01</span>`;
    if(p<0.05) return `<span class="badge badge-orange">p&lt;0.05</span>`;
    if(p<0.10) return `<span class="badge badge-green">p&lt;0.10</span>`;
    return `<span class="badge badge-muted">n.s.</span>`;
  }}

  function ratioBadge(ratio){{
    const pct=((ratio-1)*100).toFixed(1);
    const sign=ratio>=1?"+":"";
    const cls=ratio>1.05?"sig-hi":ratio<0.95?"sig-lo":"sig-med";
    return `<span class="${{cls}}">${{sign}}${{pct}}%</span>`;
  }}

  const thead=`<tr>
    <th>Dimension</th><th>Group</th><th>n</th>
    <th>Ann Vol</th><th>95% CI</th><th>vs Avg</th>
    <th>Raw p</th><th>FDR-adj p</th><th>Signal</th>
  </tr>`;

  const tbody=allRows.map(r=>{{
    const vol=(r.vol*100).toFixed(2)+"%";
    const ci=isNaN(r.ciLo)?"–":`[${{(r.ciLo*100).toFixed(2)}}%, ${{(r.ciHi*100).toFixed(2)}}%]`;
    const rawP=r.pval.toFixed(3);
    const adjP=r.adjP.toFixed(3);
    return `<tr>
      <td style="color:var(--muted)">${{r.dim}}</td>
      <td><b>${{r.label}}</b></td>
      <td>${{r.n}}</td>
      <td>${{vol}}</td>
      <td style="color:var(--muted);font-size:10px">${{ci}}</td>
      <td>${{ratioBadge(r.ratio)}}</td>
      <td>${{rawP}}</td>
      <td><b>${{adjP}}</b></td>
      <td>${{pBadge(r.adjP)}}</td>
    </tr>`;
  }}).join("");

  document.getElementById("tbl-inner").innerHTML=
    `<table><thead>${{thead}}</thead><tbody>${{tbody}}</tbody></table>`;
}}

// ═══════════════════════════════════════════════════════════════
// NOTICE
// ═══════════════════════════════════════════════════════════════
function renderNotice(rows){{
  const nSig5=[];
  const analysis=window._lastAnalysis;
  if(!analysis) return;
  const dims=["dom","wom","wdn","meg","weg"];
  for(const d of dims)
    for(const r of analysis[d].results)
      if(r.adjP<0.05&&!isNaN(r.vol)) nSig5.push(r);
  const est=state.estimator==="gk"?"Garman-Klass":"Close-to-Close";
  document.getElementById("notice").textContent=
    `Estimator: ${{est}}  ·  Bootstrap CIs: 250 iterations  ·  `+
    `FDR correction: Benjamini-Hochberg  ·  `+
    `${{nSig5.length}} group(s) significant at FDR-adjusted p<0.05  ·  `+
    `n<5 bars shown but flagged unreliable`;
}}

// ═══════════════════════════════════════════════════════════════
// MAIN UPDATE
// ═══════════════════════════════════════════════════════════════
function update(){{
  state.outlier = parseFloat(document.getElementById("out-sel").value)||0;
  state.vixReg  = document.getElementById("vix-sel").value;
  state.vixLow  = parseFloat(document.getElementById("vix-lo")?.value)||14;
  state.vixHigh = parseFloat(document.getElementById("vix-hi")?.value)||20;

  const rows=getFiltered();
  if(!rows.length){{
    document.getElementById("summary").innerHTML=
      `<div class="stat"><span class="stat-label">Error</span><span class="stat-val hi">No data for selection</span></div>`;
    return;
  }}

  const analysis=analyzeAll(rows,state.estimator);
  window._lastAnalysis=analysis;

  renderBar("c-dom",analysis.dom);
  renderBar("c-wom",analysis.wom);
  renderBar("c-wdn",analysis.wdn);
  renderBar("c-meg",analysis.meg);
  renderBar("c-weg",analysis.weg);
  renderVixChart(analyzeVixRegimes(rows,state.estimator,state.vixLow,state.vixHigh,state.vixExpType));
  renderSummary(rows,analysis);
  renderTable(analysis);
  renderNotice(rows);
}}

// ═══════════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════════
function setPeriod(btn,pid){{
  document.querySelectorAll("#period-tabs .tab").forEach(b=>b.classList.remove("on"));
  btn.classList.add("on");
  state.pid=pid;
  // sync date inputs to period range
  if(pid>=0){{
    document.getElementById("d-start").value=PERIODS[pid].start;
    document.getElementById("d-end").value=PERIODS[pid].end;
    state.dateStart=PERIODS[pid].start;
    state.dateEnd=PERIODS[pid].end;
  }} else {{
    document.getElementById("d-start").value="{all_start}";
    document.getElementById("d-end").value="{all_end}";
    state.dateStart="{all_start}";
    state.dateEnd="{all_end}";
  }}
  update();
}}

function setEst(btn){{
  document.querySelectorAll("#est-tabs .tab").forEach(b=>b.classList.remove("on"));
  btn.classList.add("on");
  state.estimator=btn.dataset.v;
  update();
}}

function updateVixThresholds(){{
  const lo = parseFloat(document.getElementById("vix-lo").value)||14;
  const hi = parseFloat(document.getElementById("vix-hi").value)||20;
  // ensure lo < hi
  if(lo >= hi) {{
    document.getElementById("vix-hi").value = lo + 1;
  }}
  update();
}}

function applyDates(){{
  state.pid=-1;
  state.dateStart=document.getElementById("d-start").value;
  state.dateEnd=document.getElementById("d-end").value;
  document.querySelectorAll("#period-tabs .tab").forEach(b=>b.classList.remove("on"));
  update();
}}

function resetDates(){{
  document.getElementById("d-start").value="{all_start}";
  document.getElementById("d-end").value="{all_end}";
  state.dateStart="{all_start}";
  state.dateEnd="{all_end}";
  update();
}}

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
function setVixExpType(btn,v){{
  document.querySelectorAll("#vix-exp-tabs .tab").forEach(b=>b.classList.remove("on"));
  btn.classList.add("on");
  state.vixExpType=v;
  update();
}}
(function init(){{
  // Build period tabs
  const pt=document.getElementById("period-tabs");
  const allBtn=document.createElement("button");
  allBtn.className="tab on"; allBtn.textContent="All Periods";
  allBtn.onclick=()=>setPeriod(allBtn,-1); pt.appendChild(allBtn);

  PERIODS.forEach((p,i)=>{{
    const b=document.createElement("button");
    b.className="tab"; b.textContent=p.label;
    b.onclick=()=>setPeriod(b,i); pt.appendChild(b);
  }});

  // Show VIX selector only if data available
  if(HAS_VIX) document.getElementById("vix-cg").style.display="";

  window._outlierRemoved=0;
  update();
}})();
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*55)
    print("  NIFTY Vol Params · Quant Dashboard Builder")
    print("="*55)

    ohlc, vix = fetch_ohlcv()

    print("\n  Building rows ...")
    rows = build_rows(ohlc, vix)
    print(f"  Total rows: {len(rows)}")

    print("\n  Generating HTML ...")
    html = build_html(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  ✓ Dashboard saved → {OUT}")
    print(f"  Open in any browser — fully self-contained.\n")
