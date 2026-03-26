#!/usr/bin/env python3
"""
NIFTY Volatility Parameter Dashboard — Advanced Quant Edition v2
IMPROVEMENTS v2:
  • Bootstrap 1000 iterations (was 250) → stable CIs
  • Minimum n≥10 filter (was n≥3) → no junk bars
  • Permutation test replaces Welch t-test → distribution-free
  • Cohen's d effect size added → practical significance
  • Skewness per group added → tail risk awareness
  • Lag-1 autocorrelation per group → CI reliability flag
  • CI overlap test vs average → stronger evidence check
  • Raw p column hidden → only FDR-adj p shown
  • Period stability check → signal in 2+/3 periods = robust
  • Out-of-sample hold-out (last 20%) → anti data-mining
  • Summary bar: Skew + Kurtosis of full dataset added
  • Reliability score per row (0–5 stars) in table
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, calendar, os
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# USER CONFIG
# ═══════════════════════════════════════════════════════════════
PERIODS = [
    dict(label="2015–2019 · Thu Expiry",   start="2015-01-01", end="2019-12-31",
         monthly_exp_dow=3, weekly_exp_dow=3),
    dict(label="2020–Oct 2024 · Thu Expiry", start="2020-01-01", end="2024-10-31",
         monthly_exp_dow=3, weekly_exp_dow=3),
    dict(label="Nov 2024–Now · Tue Expiry", start="2024-11-01", end="2026-03-25",
         monthly_exp_dow=1, weekly_exp_dow=1),
]
NIFTY_CSV = os.path.join("data", "nifty 10 year data.csv")
VIX_CSV   = os.path.join("data", "indiavix 10 year data.csv")
OUT = "nifty_vol_dashboard.html"
ANN = 252
MIN_N = 10          # minimum observations for a bar to be shown
N_BOOT = 1000       # bootstrap iterations
N_PERM = 1000       # permutation test iterations
OOS_FRAC = 0.20     # last 20% held out as out-of-sample

# ═══════════════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════════════
WD_LABEL = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

def load_csv_ohlcv(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%b-%y")
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index).normalize()
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",",""), errors="coerce")
    return df.sort_index()

def fetch_ohlcv():
    print(f"  Loading NIFTY data from {NIFTY_CSV} ...")
    if not os.path.exists(NIFTY_CSV):
        sys.exit(f"ERROR: File not found: {NIFTY_CSV}")
    raw = load_csv_ohlcv(NIFTY_CSV)
    if raw.empty: sys.exit("ERROR: NIFTY CSV returned empty data")
    print(f"  {len(raw)} rows [{raw.index[0].date()} -> {raw.index[-1].date()}]")
    vix = pd.Series(dtype=float, name="vix")
    if os.path.exists(VIX_CSV):
        try:
            vdf = load_csv_ohlcv(VIX_CSV)
            vix = vdf["Close"].squeeze().astype(float)
            vix.name = "vix"
            print(f"  {len(vix)} VIX rows")
        except Exception as ex:
            print(f"  VIX load failed ({ex})")
    return raw, vix

# ═══════════════════════════════════════════════════════════════
# TAGGING
# ═══════════════════════════════════════════════════════════════
def tag_monthly_expiry(idx, exp_dow):
    tags = []
    for ts in idx:
        y, m = ts.year, ts.month
        last_day = calendar.monthrange(y, m)[1]
        expiry = None
        for d in range(last_day, 0, -1):
            if pd.Timestamp(y, m, d).weekday() == exp_dow:
                expiry = pd.Timestamp(y, m, d); break
        if expiry is None: tags.append("")
        elif ts == expiry: tags.append("ExpiryDay")
        elif ts == expiry - pd.Timedelta(days=1): tags.append("DayBefore")
        elif ts == expiry + pd.Timedelta(days=1): tags.append("DayAfter")
        else: tags.append("")
    return tags

def tag_weekly_expiry(idx, exp_dow):
    tags = []
    for ts in idx:
        wd = ts.weekday()
        if wd == exp_dow: tags.append("ExpiryDay")
        elif (wd - exp_dow) % 7 == 1: tags.append("DayAfter")
        elif (exp_dow - wd) % 7 == 1: tags.append("DayBefore")
        else: tags.append("")
    return tags

# ═══════════════════════════════════════════════════════════════
# BUILD ROWS  (adds oos flag)
# ═══════════════════════════════════════════════════════════════
def build_rows(ohlc, vix):
    rows = []
    for pid, p in enumerate(PERIODS):
        s, e = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
        df = ohlc.loc[s:e].copy()
        if len(df) < 30:
            print(f"  Period {pid}: only {len(df)} rows — skipped"); continue
        df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna(subset=["ret"])
        ln_hl = np.log(df["High"] / df["Low"])
        ln_co = np.log(df["Close"] / df["Open"])
        df["gk"] = (0.5*ln_hl**2 - (2*np.log(2)-1)*ln_co**2).clip(lower=0)
        idx = df.index
        df["dom"] = idx.day
        df["wom"] = ((idx.day - 1) // 7) + 1
        df["wdn"] = [WD_LABEL[d] for d in idx.weekday]
        df["meg"] = tag_monthly_expiry(idx, p["monthly_exp_dow"])
        df["weg"] = tag_weekly_expiry(idx,  p["weekly_exp_dow"])
        df["vix"] = vix.reindex(idx, method="ffill").fillna(0) if len(vix) else 0.0
        df["pid"] = pid
        # out-of-sample flag: last OOS_FRAC of this period = oos
        cutoff_idx = int(len(df) * (1 - OOS_FRAC))
        df["oos"] = False
        df.iloc[cutoff_idx:, df.columns.get_loc("oos")] = True
        for ts, row in df.iterrows():
            rows.append({
                "date": ts.strftime("%Y-%m-%d"),
                "ret":  round(float(row.ret), 6),
                "gk":   round(float(row.gk),  8),
                "dom":  int(row.dom),
                "wom":  int(row.wom),
                "wdn":  row.wdn,
                "meg":  row.meg,
                "weg":  row.weg,
                "vix":  round(float(row.vix), 2),
                "pid":  int(pid),
                "oos":  bool(row.oos),
            })
        print(f"  Period {pid} ({p['label']}): {len(df)} rows "
              f"(IS={cutoff_idx}, OOS={len(df)-cutoff_idx})")
    return rows

# ═══════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════
def build_html(rows):
    periods_js = json.dumps([
        {"label":p["label"],"start":p["start"],"end":p["end"]} for p in PERIODS])
    data_js  = json.dumps(rows, separators=(',',':'))
    has_vix  = json.dumps(any(r["vix"] > 0 for r in rows))
    all_start = min(r["date"] for r in rows)
    all_end   = max(r["date"] for r in rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIFTY Vol Params · Quant Dashboard v2</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{{
  --bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--border:#30363d;
  --fg:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;
  --orange:#f0883e;--red:#ff7b72;--purple:#bc8cff;--yellow:#e3b341;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font-family:'SF Mono','Fira Code',ui-monospace,monospace;font-size:12px;min-height:100vh}}
#hdr{{background:var(--panel);border-bottom:1px solid var(--border);padding:14px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
#hdr h1{{font-size:15px;font-weight:700;letter-spacing:.5px}}
#hdr span{{font-size:11px;color:var(--muted)}}
.dot{{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;margin-right:6px;box-shadow:0 0 6px var(--green)}}
#ctrl{{background:var(--panel2);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end}}
.cg{{display:flex;flex-direction:column;gap:5px}}
.cg label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}}
.tabs{{display:flex;gap:4px;flex-wrap:wrap}}
.tab{{background:var(--panel);border:1px solid var(--border);color:var(--muted);padding:5px 11px;border-radius:5px;cursor:pointer;font-size:11px;font-family:inherit;transition:all .15s}}
.tab:hover{{border-color:var(--accent);color:var(--fg)}}
.tab.on{{background:var(--accent);border-color:var(--accent);color:#0d1117;font-weight:700}}
select,input[type=date],input[type=number]{{background:var(--panel);border:1px solid var(--border);color:var(--fg);padding:5px 8px;border-radius:5px;font-size:11px;font-family:inherit;outline:none;cursor:pointer}}
select:focus,input:focus{{border-color:var(--accent)}}
.sep{{width:1px;background:var(--border);margin:0 4px;align-self:stretch}}
#apply-btn{{background:var(--accent);color:#0d1117;border:none;padding:6px 14px;border-radius:5px;font-size:11px;font-family:inherit;font-weight:700;cursor:pointer}}
#apply-btn:hover{{opacity:.85}}
/* summary */
#summary{{background:var(--panel);border-bottom:1px solid var(--border);padding:8px 20px;display:flex;flex-wrap:wrap;gap:24px}}
.stat{{display:flex;flex-direction:column;gap:2px}}
.stat-label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}}
.stat-val{{font-size:14px;font-weight:700}}
.hi{{color:var(--orange)}}.lo{{color:var(--green)}}.neu{{color:var(--fg)}}
/* oos banner */
#oos-bar{{background:#bc8cff18;border-bottom:1px solid #bc8cff40;padding:5px 20px;font-size:10px;color:var(--purple);display:none}}
/* charts */
#charts{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:14px 16px}}
.chart-card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px;overflow:hidden}}
.chart-card h3{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;padding:0 4px}}
@media(max-width:1100px){{#charts{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:700px){{#charts{{grid-template-columns:1fr}}}}
/* table */
#tbl-wrap{{padding:0 16px 20px}}
#tbl-wrap h2{{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;padding:12px 4px 8px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:8px;overflow:hidden;font-size:11px}}
th{{background:var(--panel2);color:var(--muted);padding:8px 12px;text-align:left;font-weight:600;border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}}
th:hover{{color:var(--fg)}}
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
.badge-purple{{background:#bc8cff20;color:var(--purple);border:1px solid #bc8cff40}}
.stars{{color:var(--yellow);letter-spacing:-1px}}
.ci-overlap{{color:var(--green);font-size:10px}}
.ci-nolap{{color:var(--red);font-size:10px}}
#notice{{font-size:10px;color:var(--muted);padding:4px 20px 10px;font-style:italic}}
/* legend */
.legend-row{{display:flex;flex-wrap:wrap;gap:14px;padding:6px 20px 2px;font-size:10px;color:var(--muted)}}
.leg-item{{display:flex;align-items:center;gap:5px}}
.leg-dot{{width:10px;height:10px;border-radius:2px;display:inline-block}}
</style>
</head>
<body>

<!-- HEADER -->
<div id="hdr">
  <h1><span class="dot"></span>NIFTY Vol Params</h1>
  <span>Quant Dashboard v2 · Calendar-Segmented Realized Volatility · Statistically Hardened</span>
</div>

<!-- CONTROLS -->
<div id="ctrl">
  <div class="cg">
    <label>Period</label>
    <div class="tabs" id="period-tabs"></div>
  </div>
  <div class="sep"></div>
  <div class="cg">
    <label>Mode</label>
    <div class="tabs" id="oos-tabs">
      <button class="tab on" data-v="is"   onclick="setOOS(this)">In-Sample</button>
      <button class="tab"    data-v="oos"  onclick="setOOS(this)">Out-of-Sample</button>
      <button class="tab"    data-v="all"  onclick="setOOS(this)">Full Data</button>
    </div>
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
             style="width:52px" onchange="updateVixThresholds()" title="Low/Med boundary">
      <span style="color:var(--muted);font-size:10px">–</span>
      <input type="number" id="vix-hi" value="20" min="1" max="100" step="1"
             style="width:52px" onchange="updateVixThresholds()" title="Med/High boundary">
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

<!-- OOS BANNER -->
<div id="oos-bar">
  ⚠ Out-of-Sample mode active — showing last 20% of each period. 
  Signals here were NOT used to build the model. Green = pattern held OOS.
</div>

<!-- SUMMARY BAR -->
<div id="summary"></div>
<div id="notice"></div>

<!-- COLOR LEGEND -->
<div class="legend-row">
  <span style="color:var(--muted);font-size:10px;font-weight:600">BAR COLOR:</span>
  <span class="leg-item"><span class="leg-dot" style="background:#ff7b72"></span>Sig HIGH vol (p&lt;0.01)</span>
  <span class="leg-item"><span class="leg-dot" style="background:#f0883e"></span>Sig HIGH vol (p&lt;0.05)</span>
  <span class="leg-item"><span class="leg-dot" style="background:#3fb950"></span>Sig LOW vol (p&lt;0.01)</span>
  <span class="leg-item"><span class="leg-dot" style="background:#79c0ff"></span>Sig LOW vol (p&lt;0.05)</span>
  <span class="leg-item"><span class="leg-dot" style="background:#58a6ff"></span>Not significant</span>
  <span style="margin-left:8px;color:var(--muted)">|</span>
  <span class="leg-item" style="color:var(--muted)">Grey lines = 95% Bootstrap CI (1000 iter)</span>
  <span class="leg-item" style="color:var(--orange)">― Orange dashed = overall average</span>
</div>

<!-- CHARTS -->
<div id="charts">
  <div class="chart-card"><h3>Day of Month</h3><div id="c-dom" style="height:290px"></div></div>
  <div class="chart-card"><h3>Week of Month</h3><div id="c-wom" style="height:290px"></div></div>
  <div class="chart-card"><h3>Day of Week</h3><div id="c-wdn" style="height:290px"></div></div>
  <div class="chart-card"><h3>Monthly Expiry Window</h3><div id="c-meg" style="height:290px"></div></div>
  <div class="chart-card"><h3>Weekly Expiry Window</h3><div id="c-weg" style="height:290px"></div></div>
  <div class="chart-card"><h3>VIX Regime · Expiry Vol</h3><div id="c-vix" style="height:290px"></div></div>
</div>

<!-- TABLE -->
<div id="tbl-wrap">
  <h2>Significance Analysis — All Calendar Groups</h2>
  <div style="font-size:10px;color:var(--muted);padding:0 4px 8px">
    Sorted by Reliability Score. Click column headers to re-sort.
    ★ = reliability stars (max 5): n≥10, CI no-overlap avg, Cohen's d≥0.2, perm p&lt;0.05, stable across periods.
  </div>
  <div id="tbl-inner"></div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════
const PERIODS   = {periods_js};
const RAW       = {data_js};
const HAS_VIX   = {has_vix};
const ANN       = {ANN};
const MIN_N     = {MIN_N};
const N_BOOT    = {N_BOOT};
const N_PERM    = {N_PERM};

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════
let state = {{
  pid: -1, estimator:"cc", outlier:2.5,
  vixReg:"all", vixLow:14, vixHigh:20,
  dateStart:"{all_start}", dateEnd:"{all_end}",
  oosMode:"is",   // is | oos | all
}};
let _sortCol="stars", _sortAsc=false;

// ═══════════════════════════════════════════════════════════════
// STATS ENGINE
// ═══════════════════════════════════════════════════════════════
function mean(a){{ return a.length?a.reduce((s,v)=>s+v,0)/a.length:0 }}
function sampleVar(a){{
  const n=a.length; if(n<2) return 0;
  const mu=mean(a); return a.reduce((s,v)=>s+(v-mu)**2,0)/(n-1);
}}
function skewness(a){{
  const n=a.length; if(n<3) return 0;
  const mu=mean(a), sd=Math.sqrt(sampleVar(a));
  if(sd<1e-12) return 0;
  return a.reduce((s,v)=>s+((v-mu)/sd)**3,0)/n;
}}
function kurtosis(a){{
  const n=a.length; if(n<4) return 0;
  const mu=mean(a), sd=Math.sqrt(sampleVar(a));
  if(sd<1e-12) return 0;
  return a.reduce((s,v)=>s+((v-mu)/sd)**4,0)/n - 3; // excess
}}
function autocorr1(a){{
  if(a.length<4) return 0;
  const mu=mean(a), n=a.length;
  let num=0, den=0;
  for(let i=0;i<n-1;i++) num+=(a[i]-mu)*(a[i+1]-mu);
  for(let i=0;i<n;i++)   den+=(a[i]-mu)**2;
  return den<1e-12?0:num/den;
}}
function cohenD(g1,g2){{
  if(g1.length<2||g2.length<2) return 0;
  const a1=g1.map(Math.abs), a2=g2.map(Math.abs);
  const n1=a1.length, n2=a2.length;
  const pooledSD=Math.sqrt(((n1-1)*sampleVar(a1)+(n2-1)*sampleVar(a2))/(n1+n2-2));
  return pooledSD<1e-12?0:Math.abs(mean(a1)-mean(a2))/pooledSD;
}}
function annVolCC(rets){{ return Math.sqrt(Math.max(sampleVar(rets),0)*ANN) }}
function annVolGK(gks){{  return Math.sqrt(Math.max(mean(gks),0)*ANN) }}
function estVol(vals,est){{ return est==="gk"?annVolGK(vals):annVolCC(vals) }}

// 1000-iteration bootstrap CI
function bootstrap(vals,est,nIter=N_BOOT){{
  const n=vals.length;
  if(n<4) return [NaN,NaN];
  const boots=[];
  for(let i=0;i<nIter;i++){{
    const s=[];
    for(let j=0;j<n;j++) s.push(vals[Math.floor(Math.random()*n)]);
    boots.push(estVol(s,est));
  }}
  boots.sort((a,b)=>a-b);
  return [boots[Math.floor(0.025*nIter)], boots[Math.floor(0.975*nIter)]];
}}

// Permutation test (distribution-free) on |returns|
function permTest(g1,g2,nPerm=N_PERM){{
  if(g1.length<3||g2.length<3) return 1;
  const a1=g1.map(Math.abs), a2=g2.map(Math.abs);
  const obs=Math.abs(mean(a1)-mean(a2));
  const combined=[...a1,...a2];
  const n1=a1.length;
  let count=0;
  for(let i=0;i<nPerm;i++){{
    // Fisher-Yates shuffle
    for(let j=combined.length-1;j>0;j--){{
      const k=Math.floor(Math.random()*(j+1));
      [combined[j],combined[k]]=[combined[k],combined[j]];
    }}
    const d=Math.abs(mean(combined.slice(0,n1))-mean(combined.slice(n1)));
    if(d>=obs) count++;
  }}
  return count/nPerm;
}}

function bhFDR(pvals){{
  const n=pvals.length;
  const idx=pvals.map((p,i)=>({{p,i}})).sort((a,b)=>a.p-b.p);
  const adj=new Array(n).fill(1);
  for(let k=0;k<n;k++) adj[idx[k].i]=Math.min(1,idx[k].p*n/(k+1));
  for(let k=n-2;k>=0;k--) adj[idx[k].i]=Math.min(adj[idx[k].i],adj[idx[k+1].i]);
  return adj;
}}

// ═══════════════════════════════════════════════════════════════
// FILTERING
// ═══════════════════════════════════════════════════════════════
function getFiltered(){{
  let rows=RAW.filter(r=>{{
    if(state.pid>=0 && r.pid!==state.pid) return false;
    if(r.date<state.dateStart||r.date>state.dateEnd) return false;
    if(state.oosMode==="is"  && r.oos) return false;
    if(state.oosMode==="oos" && !r.oos) return false;
    return true;
  }});
  if(state.outlier>0 && rows.length>5){{
    const rets=rows.map(r=>r.ret);
    const mu=mean(rets), sd=Math.sqrt(sampleVar(rets));
    const thr=state.outlier*sd, before=rows.length;
    rows=rows.filter(r=>Math.abs(r.ret-mu)<=thr);
    window._outlierRemoved=before-rows.length;
  }} else {{ window._outlierRemoved=0; }}
  if(state.vixReg!=="all" && rows.some(r=>r.vix>0)){{
    const lo=state.vixLow, hi=state.vixHigh;
    if(state.vixReg==="low")  rows=rows.filter(r=>r.vix>0&&r.vix<lo);
    if(state.vixReg==="med")  rows=rows.filter(r=>r.vix>0&&r.vix>=lo&&r.vix<=hi);
    if(state.vixReg==="high") rows=rows.filter(r=>r.vix>0&&r.vix>hi);
  }}
  return rows;
}}

// ═══════════════════════════════════════════════════════════════
// PERIOD STABILITY: how many periods show same signal direction
// ═══════════════════════════════════════════════════════════════
function periodStability(allRows, keyFn, lbl, estimator){{
  let sigCount=0, totalPeriods=0;
  for(let pid=0;pid<PERIODS.length;pid++){{
    const prows=allRows.filter(r=>r.pid===pid);
    if(prows.length<20) continue;
    totalPeriods++;
    const grp=prows.filter(r=>keyFn(r)===lbl);
    const rest=prows.filter(r=>keyFn(r)!==lbl);
    if(grp.length<MIN_N) continue;
    const gVals=estimator==="gk"?grp.map(r=>r.gk):grp.map(r=>r.ret);
    const allV=estimator==="gk"?prows.map(r=>r.gk):prows.map(r=>r.ret);
    const ratio=estVol(gVals,estimator)/estVol(allV,estimator);
    const p=permTest(grp.map(r=>r.ret), rest.map(r=>r.ret), 200);
    if(p<0.10) sigCount++;
  }}
  return {{sigCount, totalPeriods}};
}}

// ═══════════════════════════════════════════════════════════════
// ANALYSIS
// ═══════════════════════════════════════════════════════════════
function analyzeGroup(rows, keyFn, labels, estimator){{
  const allVals=estimator==="gk"?rows.map(r=>r.gk):rows.map(r=>r.ret);
  const overallVol=estVol(allVals,estimator);
  const results=[];

  for(const lbl of labels){{
    const grp =rows.filter(r=>keyFn(r)===lbl);
    const rest=rows.filter(r=>keyFn(r)!==lbl);
    const gVals=estimator==="gk"?grp.map(r=>r.gk):grp.map(r=>r.ret);

    if(grp.length<MIN_N){{
      results.push({{label:String(lbl),n:grp.length,vol:NaN,ciLo:NaN,ciHi:NaN,
        pval:1,adjP:1,ratio:NaN,meanRet:NaN,cohend:NaN,skew:NaN,ac1:NaN,
        ciOverlap:true,stab:{{sigCount:0,totalPeriods:0}},stars:0}});
      continue;
    }}

    const vol=estVol(gVals,estimator);
    const [ciLo,ciHi]=bootstrap(gVals,estimator,N_BOOT);
    const pval=permTest(grp.map(r=>r.ret), rest.map(r=>r.ret), N_PERM);
    const mr=mean(grp.map(r=>r.ret));
    const cd=cohenD(grp.map(r=>r.ret), rest.map(r=>r.ret));
    const sk=skewness(grp.map(r=>r.ret));
    const ac=autocorr1(grp.map(r=>r.ret));
    const ratio=vol/overallVol;
    // CI overlap with overall average
    const ciOverlap=!(ciHi<overallVol||ciLo>overallVol);
    const stab=periodStability(RAW.filter(r=>r.date>=state.dateStart&&r.date<=state.dateEnd),
                                keyFn,lbl,estimator);

    results.push({{label:String(lbl),n:grp.length,vol,ciLo,ciHi,
      pval,adjP:1,ratio,meanRet:mr,cohend:cd,skew:sk,ac1:ac,
      ciOverlap,stab,stars:0}});
  }}

  // BH-FDR on permutation p-values
  const adjP=bhFDR(results.map(r=>r.pval));
  results.forEach((r,i)=>{{
    r.adjP=adjP[i];
    // Reliability stars (0-5):
    // 1) n>=MIN_N, 2) CI does NOT overlap avg, 3) Cohen's d>=0.2,
    // 4) FDR-adj p<0.05, 5) stable in 2+ periods
    let s=0;
    if(r.n>=MIN_N)            s++;
    if(!r.ciOverlap)          s++;
    if(r.cohend>=0.2)         s++;
    if(r.adjP<0.05)           s++;
    if(r.stab.sigCount>=2)    s++;
    r.stars=s;
  }});
  return {{results,overallVol}};
}}

function analyzeAll(rows,estimator){{
  const domAll=[...Array(31)].map((_,i)=>i+1);
  const domResult=analyzeGroup(rows,r=>r.dom,domAll,estimator);
  domResult.results=domResult.results.filter(r=>r.n>=MIN_N||isNaN(r.vol));
  const wom=analyzeGroup(rows,r=>r.wom,[1,2,3,4,5],estimator);
  const wdn=analyzeGroup(rows,r=>r.wdn,["Mon","Tue","Wed","Thu","Fri"],estimator);
  const meg=analyzeGroup(rows,r=>r.meg,["DayBefore","ExpiryDay","DayAfter"],estimator);
  const weg=analyzeGroup(rows,r=>r.weg,["DayBefore","
