from __future__ import annotations
import shutil
import time
# pk4_northern_star_app_2026-02-04_v41.py
# Streamlit app: Pick 4 "Northern Star" core stream ranking + Rare/Ultra-Rare engines (AAAB+AABB, AAAA)
# Notes:
# - Designed to work with LotteryPost-style exports (tab .txt or .csv) that include Date, State, Game, Results.
# - Ignores Wild Ball / Fireball / multipliers by extracting the first 4 digits like "1-2-3-4".
# - Excludes Maryland by default (toggle in sidebar).

def _parse_date_series(s: 'pd.Series') -> 'pd.Series':
    """Robustly parse a Date-like Series into pandas datetime (normalized).
    Handles ints like 20250131, strings '2025-01-31', and timestamps."""
    try:
        if s is None:
            return s
        # Fast path: already datetime-like
        if np.issubdtype(getattr(s.dtype, 'type', object), np.datetime64):
            return pd.to_datetime(s, errors='coerce')
        # If numeric and looks like YYYYMMDD, parse with explicit format
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            s_num = pd.to_numeric(s, errors='coerce')
            # treat as YYYYMMDD when within plausible range and no decimals
            mask = s_num.notna()
            if mask.any():
                vals = s_num[mask].astype('int64', errors='ignore')
                # Heuristic: 8-digit dates between 19000101 and 21000101
                if ((vals >= 19000101) & (vals <= 21000101)).mean() > 0.8:
                    s_str = s_num.astype('Int64').astype(str)
                    out = pd.to_datetime(s_str, format='%Y%m%d', errors='coerce')
                    return out
        # Otherwise try string parse; if most are 8-digit, use YYYYMMDD
        s_str = s.astype(str)
        digit8 = s_str.str.fullmatch(r"\d{8}").fillna(False)
        if digit8.mean() > 0.8:
            return pd.to_datetime(s_str, format='%Y%m%d', errors='coerce')
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return pd.to_datetime(s, errors='coerce')



APP_VERSION = "v135 (Member Traits Label Fix + Validator)"

CHANGE_LOG_V51 = """v51 — SeedTraits + Cadence + AllCores Cache-Only (built from v50, NO regressions)

✅ Added:
- Seed Traits (positive + negative) autoload + optional upload; soft scoring applied to:
  - Northern Lights UniversalScore
  - Core scoring (Northern Star + Core View helper)
- Cadence scoring (windowed 180/365) as a soft, configurable boost (no hard filters)
- Northern Star tab (restores Rare Engine + Ultra-Rare engine outputs in UI)
- Global all-cores RankPos percentile map (cache-only) + per-core maps remain distinct
- “Select all cores” button for multi-core selection (cache building / batch tools)

✅ Fixed (signature/return + robustness):
- Added PctStrength alias to percentile map output (back-compat)
- Northern Lights: position strength now resolves via RankPos (not Stream) to avoid empty maps
- Bucket recommendations now include back-compat metadata keys (top_n / due_ranks / etc)
- All Cores mode in Northern Lights is now STRICT cache-only; refuses live compute if any core cache missing

Notes:
- No sections removed or disabled. New functionality is additive and defaults are conservative.
"""


# Core presets (family IDs) shown in the UI. Keep this list additive.
# These are the cores you and I have explicitly worked on so far.
CORE_PRESETS = [
    "012",
    "013",
    "016",
    "017",
    "018",
    "019",
    "023",
    "024",
    "025",
    "027",
    "028",
    "029",
    "035",
    "038",
    "046",
    "048",
    "056",
    "059",
    "067",
    "068",
    "078",
    "129",
    "134",
    "135",
    "145",
    "146",
    "149",
    "167",
    "168",
    "169",
    "178",
    "179",
    "236",
    "238",
    "239",
    "245",
    "246",
    "249",
    "257",
    "258",
    "278",
    "279",
    "345",
    "348",
    "357",
    "358",
    "359",
    "378",
    "379",
    "389",
    "456",
    "457",
    "458",
    "459",
    "468",
    "479",
    "489",
    "567",
    "568",
    "579",
    "589",
    "679",
    "689",
    "789",
]


# Compatibility: the legacy 'old app' core set (kept for quick selection)
OLD_APP_CORE_SET = ['016', '017', '018', '019', '023', '024', '025', '027', '028', '029', '038', '046', '048', '056', '059', '067', '068', '078', '129', '135', '145', '146', '149', '167', '168', '169', '179', '236', '238', '239', '245', '246', '249', '257', '258', '278', '279', '345', '348', '357', '359', '378', '379', '389', '457', '459', '489', '567', '579', '589', '679', '689', '789']
# --- Optional: "Trigger Map" weighting for a fixed 39-play list (soft boost, never an elimination) ---
# This is intentionally conservative: it only adds a small score nudge to prioritize certain plays per-stream
# based on the previous winner in that same stream.
TRIGGER_PLAYLIST_39 = [
    "3389","3889","3899",
    "0013","0113","0133","0019","0119","0199",
    "1145","1445","1455","1147","1447","1477","1149","1499","1449",
    "1136","1336","1366",
    "1667","1167","1677","1169","1669","1699",
    "3356","3566","3556","3367","3667","3677",
    "5567","5667","5677","6679","6779","6799",
]

# Override trigger: previous winner contains >=3 digits from {7,8,9,0}
_TRIGGER_OVERRIDE_SET = set("7890")

# Default decision tree: bucket by last digit of previous winner
TRIGGER_BY_PREV_LAST = {
    "0": ["3556","3899","1677","5677","3677"],
    "1": ["5567","0113","1499","1699","1167"],
    "2": ["1699","1677","3566","1667","1149"],
    "3": ["6679","1167","0019","3566","1669"],
    "4": ["1366","1667","3356","1455"],
    "5": ["0133","1147","1136","1445","1145"],
    "6": ["1449","0199","3356","3367","3556"],
    "7": ["3677","1145","0013","1447","1169"],
    "8": ["1366","1149","3389","1669","5667"],
    "9": ["1445","1149","6679","1669","1169"],
}

TRIGGER_OVERRIDE_EMPHASIS = ["3889","3677","1169","3899","3556","6679"]

def _safe_get(obj, key, default=None):
    # Safe .get for dict/Series/etc; falls back to __getitem__ when appropriate.
    try:
        if obj is None:
            return default
        getter = getattr(obj, 'get', None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                try:
                    return getter(key)
                except Exception:
                    return default
        try:
            return obj[key]
        except Exception:
            return default
    except Exception:
        return default
def trigger_map_emphasis(prev_result_4d: str) -> list[str]:
    """Return a (possibly empty) ordered emphasis list for the Trigger Map."""
    s = re.sub(r"\D", "", str(prev_result_4d or ""))[:4]
    if len(s) != 4:
        return []
    # Override: >=3 digits from 7/8/9/0
    if sum(1 for ch in s if ch in _TRIGGER_OVERRIDE_SET) >= 3:
        return list(TRIGGER_OVERRIDE_EMPHASIS)
    return list(TRIGGER_BY_PREV_LAST.get(s[-1], []))

def trigger_map_boost(play_4d: str, prev_result_4d: str, *, boost_points: float = 2.0) -> float:
    """Soft boost points for a play given the previous result in the stream."""
    if not play_4d:
        return 0.0
    p = re.sub(r"\D", "", str(play_4d)).zfill(4)[-4:]
    emph = trigger_map_emphasis(prev_result_4d)
    if not emph:
        return 0.0
    return float(boost_points) if p in set(emph) else 0.0


import re

def _safe_int(x):
    """Convert x to int if possible; returns None if not."""
    if x is None:
        return None
    if isinstance(x, int):
        return int(x)
    try:
        import numpy as _np
        if isinstance(x, (_np.integer,)):
            return int(x)
    except Exception:
        pass
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None
    try:
        return int(x)
    except Exception:
        return None

import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd

# --- Robustness: guard accidental `.get()` calls on pandas scalar types (Timestamp/NaT/Index) ---
# Some backtest paths treat values as dict/Series and call `.get(...)`. Under certain data
# shapes (e.g., scalar Timestamp/NaT sneaking through), this can raise AttributeError.
# We defensively add a `.get` method to common pandas scalar/index types to return default.
try:
    from pandas._libs.tslibs.timestamps import Timestamp as _PandasTimestamp  # type: ignore
    if not hasattr(_PandasTimestamp, 'get'):
        def _ts_get(self, _key=None, _default=None):  # noqa: D401
            return _default
        _PandasTimestamp.get = _ts_get  # type: ignore[attr-defined]
except Exception:
    pass
try:
    from pandas._libs.tslibs.nattype import NaTType as _PandasNaTType  # type: ignore
    if not hasattr(_PandasNaTType, 'get'):
        def _nat_get(self, _key=None, _default=None):
            return _default
        _PandasNaTType.get = _nat_get  # type: ignore[attr-defined]
except Exception:
    pass
try:
    from pandas import Index as _PandasIndex  # type: ignore
    if not hasattr(_PandasIndex, 'get'):
        def _idx_get(self, _key=None, _default=None):
            return _default
        _PandasIndex.get = _idx_get  # type: ignore[attr-defined]
except Exception:
    pass

import traceback
# --- Compatibility guard: some legacy code paths may reference `Pandas.get(...)` or `pd.get(...)`
# Pandas (the module) does not define `.get` by default; define a safe helper to avoid crashes.
Pandas = pd
if not hasattr(Pandas, 'get'):
    def _pandas_module_get(attr, default=None):
        return getattr(Pandas, attr, default)
    Pandas.get = _pandas_module_get  # type: ignore
if not hasattr(pd, 'get'):
    pd.get = Pandas.get  # type: ignore


def _to_dataframe(obj) -> pd.DataFrame:
    """Best-effort conversion for Streamlit display; prevents 'dict has no dtype' crashes."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, dict):
        # Prefer a single-row DF for dicts
        return pd.DataFrame([obj])
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame({"value": [str(obj)]})

import streamlit as st

# --- BUILD STAMP (auto) ---
APP_BUILD_ID = "v135_MEMBER_TRAITS_LABELFIX_VALIDATOR_20260219T000000Z"
APP_BUILD_BASE = "base=v135_MEMBER_TRAITS_LABELFIX"
APP_BUILD_HASH = 'kpi_searchsorted_v1'
# --------------------------

def _bt_settings_sig(**kwargs) -> str:
    """Stable signature for backtest settings so cached results can't mask weight/trait changes."""
    import json as _json
    import hashlib as _hashlib

    # Keep only JSON-serializable primitives
    clean = {}
    for k, v in kwargs.items():
        try:
            if isinstance(v, (list, tuple, set)):
                clean[k] = [str(x) for x in v]
            elif isinstance(v, (float, int, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        except Exception:
            clean[k] = str(v)

    payload = _json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hashlib.sha256(payload).hexdigest()[:16]


# --- Safe rerun helper (Streamlit Cloud compatibility) ---
def _safe_rerun():
    """Rerun app without crashing across Streamlit versions."""
    try:
        st.rerun()  # Streamlit >= 1.18
        return
    except Exception:
        pass
    try:
        st.experimental_rerun()  # older Streamlit
        return
    except Exception:
        pass



# --- Global defaults (used by widgets across tabs) ---
# Some views (e.g., Auto Profit Planner) render before Backtest variables exist.
# Keep these simple, stable defaults to avoid NameError while preserving behavior.
if "cost_per_play" not in globals():
    cost_per_play = 0.25  # BetAnything minimum ticket
if "payout_per_win" not in globals():
    payout_per_win = 247.50  # common Pick4 straight $1-equivalent example; can be overridden by UI where applicable

# --- Safety init (prevents NameError when UI blocks are skipped) ---
member_track: bool = False

import hashlib
import datetime
import json
from functools import lru_cache

# ---- Safe defaults to prevent NameError during first render ----
cfg = None  # set later after window selection

# --- Quick-pick core groups
SPECIAL_7_CORES = ['246','168','589','019','468','236','025']

exclude_md = True  # default behavior: exclude Maryland unless user toggles off
map_file = None  # backward-compatible alias set in sidebar

# Tab containers (assigned after tabs() is created; keep placeholders to avoid NameError)
_t_nl = None
_t_ns = None
_t_core = None
_t_bt = None

# ---- Rerun helper (must be defined early; used by core-selection buttons) ----
def _rerun() -> None:
    """Compatibility rerun across Streamlit versions."""
    try:
        # Streamlit >= 1.30
        st.rerun()
        return
    except Exception:
        pass
    try:
        # Older Streamlit
        st.experimental_rerun()
        return
    except Exception:
        pass
    # Last resort: no-op (should not happen on Streamlit Cloud)
    return






# -------------------------
# Parsing + helpers
# -------------------------

# -------------------------
# Disk baseline cache (optional, keeps runs fast)
# -------------------------
from pathlib import Path as _Path

DISK_CACHE_DIR = _Path("pk4_baseline_cache")

DISK_PCT_DIR = DISK_CACHE_DIR / "pctmaps"
DISK_PCT_DIR.mkdir(parents=True, exist_ok=True)

def _pctmap_path(core: str, window_days: int) -> Path:
    safe_core = core.replace("/", "_")
    return DISK_PCT_DIR / f"rankpos_pct_{safe_core}_{window_days}d.csv"

def _save_pctmap_to_disk(core: str, window_days: int, pct_df: pd.DataFrame, asof_last_date: str) -> None:
    if pct_df is None or pct_df.empty:
        return
    out = pct_df.copy()
    out.insert(0, "core", core)
    out.insert(1, "window_days", int(window_days))
    out.insert(2, "asof_last_date", asof_last_date)
    out.to_csv(_pctmap_path(core, window_days), index=False)

def _load_pctmap_from_disk(core: str, window_days: int, expected_last_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    p = _pctmap_path(core, window_days)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if expected_last_date is not None and "asof_last_date" in df.columns:
        # Keep only if matches current history last date (prevents stale/inaccurate maps)
        if str(df["asof_last_date"].iloc[0]) != str(expected_last_date):
            return None
    return df


# --- NS stream-rank persistence (disk) ---
def _ns_stream_rank_path(window_days: int) -> Path:
    return DISK_CACHE_DIR / f"ns_stream_rank_df_{int(window_days)}.parquet"

def _save_ns_stream_rank_to_disk(window_days: int, df) -> None:
    try:
        DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _ns_stream_rank_path(window_days)
        # Prefer parquet, fall back to csv if parquet fails in the runtime env.
        try:
            df.to_parquet(p, index=False)
        except Exception:
            df.to_csv(p.with_suffix('.csv'), index=False)
    except Exception:
        # Never fail the app because disk persistence failed
        return

def _load_ns_stream_rank_from_disk(window_days: int):
    try:
        p = _ns_stream_rank_path(window_days)
        if p.exists():
            import pandas as _pd
            return _pd.read_parquet(p)
        csvp = p.with_suffix('.csv')
        if csvp.exists():
            import pandas as _pd
            return _pd.read_csv(csvp)
    except Exception:
        return None
    return None

def build_allcores_rankpos_pctmap(
    cores_list: List[str],
    window_days: int,
    expected_last_date: Optional[str],
    cache_only: bool = True,
    df_all: Optional[pd.DataFrame] = None,
    stream_rankings_df: Optional[pd.DataFrame] = None,
    family_counts_df: Optional[pd.DataFrame] = None,
    struct_counts_df: Optional[pd.DataFrame] = None,
    cfg: Optional["RankConfig"] = None,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Aggregate RankPos->HitCount across many cores and return a percentile map.

    If cache_only=True, requires baseline cache for every core; if any is missing/outdated,
    returns None (caller decides how to handle).
    """
    frames = []
    missing = []
    for core in cores_list:
        if cache_only:
            ss, _pos_df, _meta = _load_baseline_from_disk(core, window_days, expected_last_date=expected_last_date)
            if ss is None or ss.empty:
                missing.append(str(core).zfill(3))
                continue
        else:
            if cfg is None:
                cfg = RankConfig()
                cfg.window_days = window_days
            ss = compute_stream_stats(df_all, core, window_days=window_days, exclude_md=False)
        if ss is None or ss.empty or "RankPos" not in ss.columns:
            continue
        # Keep minimal columns
        if "HitsWindow" in ss.columns:
            frames.append(ss[["RankPos", "HitsWindow"]].copy())
        elif "HitCount" in ss.columns:
            tmp = ss[["RankPos", "HitCount"]].copy()
            tmp = tmp.rename(columns={"HitCount": "HitsWindow"})
            frames.append(tmp)

    if missing:
        return None, missing
    if not frames:
        return pd.DataFrame(), []

    comb = pd.concat(frames, ignore_index=True)
    # Aggregate by RankPos; position_percentile_map will also group, but we keep it clean
    comb = comb.groupby("RankPos", as_index=False)["HitsWindow"].sum()
    pct, _ = position_percentile_map(comb)
    return pct, []
DISK_CACHE_DIR.mkdir(exist_ok=True)

# -------------------------
# Rolling baseline store (optional)
# - Lets the app "self-maintain" a rolling ~3-year history by appending from the 24h file
# - Purges rows older than ~3 years from the newest date in the store
# - Stored on disk as parquet (preferred) or CSV (fallback), plus a small JSON meta file
# -------------------------
BASELINE_STORE_DIR = _Path("pk4_baseline_store")
BASELINE_STORE_DIR.mkdir(exist_ok=True)

def _hard_reset_disk_and_streamlit_caches():
    """Aggressively clears Streamlit caches + deletes on-disk baseline caches used by this app.

    Notes:
    - Streamlit reruns automatically after the button click; no explicit st.rerun needed.
    - We verify disk deletions and report if something couldn't be removed (permissions, locks, etc.).
    """
    actions = []

    # 1) Streamlit caches (memory)
    try:
        st.cache_data.clear()
        actions.append("Cleared st.cache_data")
    except Exception as e:
        actions.append(f"Could not clear st.cache_data: {type(e).__name__}: {e}")

    try:
        st.cache_resource.clear()
        actions.append("Cleared st.cache_resource")
    except Exception as e:
        actions.append(f"Could not clear st.cache_resource: {type(e).__name__}: {e}")

    # 2) On-disk caches
    # Prefer the same dirs the app writes to, but also sweep for any legacy paths.
    candidates = []
    for d in [DISK_CACHE_DIR, BASELINE_STORE_DIR]:
        if d is None:
            continue
        candidates.append(Path(d))

    # Sweep for any folders matching known cache naming (covers older versions / accidental cwd changes).
    try:
        root = Path(os.getcwd())
        for name in ["pk4_baseline_cache", "pk4_baseline_store"]:
            for p in root.rglob(name):
                candidates.append(Path(p))
        # Some environments use /tmp; safe to attempt
        tmp_root = Path("/tmp")
        if tmp_root.exists():
            for name in ["pk4_baseline_cache", "pk4_baseline_store"]:
                for p in tmp_root.rglob(name):
                    candidates.append(Path(p))
    except Exception:
        pass

    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    for d in uniq:
        try:
            if d.exists():
                shutil.rmtree(str(d), ignore_errors=True)
                if d.exists():
                    actions.append(f"WARNING: could not delete disk dir (still exists): {d}")
                else:
                    actions.append(f"Deleted disk dir: {d}")
        except Exception as e:
            actions.append(f"Could not delete disk dir {d}: {type(e).__name__}: {e}")

    # 3) Session state reset (keep only a tiny allowlist so the UI can show the reset log)
    try:
        keep_keys = {"_hard_reset_actions", "_hard_reset_nonce"}
        for k in list(st.session_state.keys()):
            if k in keep_keys:
                continue
            st.session_state.pop(k, None)
        actions.append(f"Cleared session_state keys (kept {len(keep_keys)})")
    except Exception as e:
        actions.append(f"Could not clear session_state: {type(e).__name__}: {e}")

    return actions
def _ensure_list(x):
    """Return x as a list suitable for pandas .isin()."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    # pandas Series / Index
    try:
        import pandas as _pd
        if isinstance(x, (_pd.Series, _pd.Index)):
            return x.tolist()
    except Exception:
        pass
    # scalar -> list
    return [x]

def _coerce_store_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = _parse_date_series(df["Date"])
        df = df[df["Date"].notna()].copy()
    # Ensure required columns exist (best-effort)
    for c in ["State","Game","Results","Pick4","Structure","Box","Stream"]:
        if c not in df.columns:
            df[c] = None
    return df

def load_baseline_store() -> pd.DataFrame:
    df = _safe_read_table(BASELINE_STORE_BASE)
    if df is None:
        return pd.DataFrame()
    df = _coerce_store_df(df)
    # Recompute derived fields if store was CSV without them
    if not df.empty:
        if "Pick4" not in df.columns or df["Pick4"].isna().all():
            df["Pick4"] = df.get("Results", pd.Series([None]*len(df))).map(extract_pick4_digits)
        if "Structure" not in df.columns or df["Structure"].isna().all():
            df["Structure"] = df["Pick4"].map(structure_of_4)
        if "Box" not in df.columns or df["Box"].isna().all():
            df["Box"] = df["Pick4"].map(box_key)
        if "Stream" not in df.columns or df["Stream"].isna().all():
            df["Stream"] = df["State"].astype(str).str.strip() + " | " + df["Game"].astype(str).str.strip()
        df = df[df["Pick4"].notna()].copy()
    return df

def write_baseline_store(df: pd.DataFrame, note: str = "") -> Tuple[bool, str]:
    df = _coerce_store_df(df)
    ok, path_written = _safe_write_table(df, BASELINE_STORE_BASE)
    meta = _read_meta(BASELINE_STORE_BASE)
    meta.update({
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "rows": int(df.shape[0]) if df is not None else 0,
        "max_date": str(df["Date"].max()) if df is not None and not df.empty else "",
    })
    _write_meta(BASELINE_STORE_BASE, meta)
    return ok, path_written

def purge_to_rolling_3y(df: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    df = _coerce_store_df(df)
    if df.empty:
        return df
    max_date = df["Date"].max()
    # 3-year rolling window; add a small buffer for leap years
    cutoff = pd.Timestamp(max_date) - pd.Timedelta(days=(365*years + 7))
    df2 = df[df["Date"] >= cutoff].copy()
    return df2

def append_from_24h(df_store: pd.DataFrame, df_24h: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df_store = _coerce_store_df(df_store)
    df_24h = _coerce_store_df(df_24h)
    if df_24h.empty:
        return df_store, 0
    # Only keep rows with the needed fields
    df_24h = df_24h[["Date","State","Game","Results","Pick4","Structure","Box","Stream"]].copy()
    df_store = df_store[["Date","State","Game","Results","Pick4","Structure","Box","Stream"]].copy() if not df_store.empty else df_store

    # Dedup key: Date + State + Game (unique stream-day draw)
    def _key(df):
        return (
            df["Date"].dt.strftime("%Y-%m-%d").astype(str)
            + "|" + df["State"].astype(str).str.strip().str.lower()
            + "|" + df["Game"].astype(str).str.strip().str.lower()
        )
    if df_store.empty:
        out = df_24h.copy()
        return out, int(out.shape[0])

    store_keys = set(_key(df_store).tolist())
    df_24h["_k"] = _key(df_24h)
    new_rows = df_24h[~df_24h["_k"].isin(store_keys)].drop(columns=["_k"]).copy()
    if new_rows.empty:
        return df_store, 0
    out = pd.concat([df_store, new_rows], ignore_index=True)
    # Final dedup safety
    out["_k"] = _key(out)
    out = out.drop_duplicates(subset=["_k"]).drop(columns=["_k"]).copy()
    return out, int(new_rows.shape[0])

def _cache_key(max_date: pd.Timestamp, rows: int, streams: int, exclude_md: bool, window_days: int, cores: List[str]) -> str:
    # small, stable key so your cache survives restarts
    core_sig = "-".join(cores)
    base = f"{max_date.date()}|{rows}|{streams}|md={int(exclude_md)}|w={window_days}|{core_sig}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

def _safe_write_table(df: pd.DataFrame, path: _Path) -> Tuple[bool, str]:
    """Write as parquet if possible, else as CSV (human readable)."""
    try:
        if _parquet_available():
            df.to_parquet(path.with_suffix(".parquet"), index=False)
            return True, str(path.with_suffix(".parquet"))
        df.to_csv(path.with_suffix(".csv"), index=False)
        return True, str(path.with_suffix(".csv"))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _safe_read_table(path: _Path) -> Optional[pd.DataFrame]:
    # Accept either a "base" path (no suffix) or a fully-qualified .parquet/.csv path.
    try:
        if path.suffix.lower() == ".parquet" and path.exists():
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv" and path.exists():
            return pd.read_csv(path)

        p_parq = path.with_suffix(".parquet")
        p_csv = path.with_suffix(".csv")
        if p_parq.exists():
            return pd.read_parquet(p_parq)
        if p_csv.exists():
            return pd.read_csv(p_csv)
    except Exception:
        return None
    return None


def _read_meta(path: _Path) -> Dict[str, Any]:
    p = path.with_suffix(".json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _write_meta(path: _Path, meta: Dict[str, Any]) -> None:
    p = path.with_suffix(".json")
    p.write_text(json.dumps(meta, indent=2, default=str))

FOUR_DIGITS_RE = re.compile(r"(\d)\s*-\s*(\d)\s*-\s*(\d)\s*-\s*(\d)")


def _bytes_of_upload(uploaded) -> bytes:
    if uploaded is None:
        return b""
    try:
        return uploaded.getvalue()
    except Exception:
        try:
            return uploaded.read()
        except Exception:
            return b""

def file_fingerprint(uploaded) -> str:
    """Stable fingerprint for an uploaded file (used to auto-recompute when data changes)."""
    data = _bytes_of_upload(uploaded)
    if not data:
        return ""
    return hashlib.sha1(data).hexdigest()

def most_recent_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "Date" not in df.columns:
        return None
    try:
        return pd.to_datetime(df["Date"]).max()
    except Exception:
        return None
def extract_pick4_digits(results: str) -> Optional[str]:
    """Return 4-digit string from LotteryPost 'Results' cell, else None."""
    if results is None or (isinstance(results, float) and np.isnan(results)):
        return None
    m = FOUR_DIGITS_RE.search(str(results))
    if not m:
        # Sometimes results can be plain "1234"
        m2 = re.search(r"\b(\d{4})\b", str(results))
        if m2:
            return m2.group(1)
        return None
    return "".join(m.groups())

def box_key(s: str) -> str:
    return "".join(sorted(s))


from itertools import permutations

def extract_4digit(x: Any) -> Optional[str]:
    """Best-effort normalize to a 4-digit string (used for straight permutation generation)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    # If already exactly 4 digits
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", s)
    if m:
        return m.group(1)
    # Try LotteryPost hyphenated format
    return extract_pick4_digits(s)


@lru_cache(maxsize=10000)
def unique_straights_for_box(box4: str) -> tuple[str, ...]:
    """Return unique 4-digit straight permutations for a 4-digit string (digits may repeat).

    Cached because the same box patterns repeat across streams/days.
    """
    box4 = extract_4digit(box4)
    if not box4:
        return tuple()
    return tuple(sorted({"".join(p) for p in permutations(box4, 4)}))


def top_straights_for_box_from_history(
    df_hist: pd.DataFrame,
    member_box4: str,
    stream_key: str | None = None,
    top_k: int = 3,
    min_stream_samples: int = 20,
    min_global_samples: int = 50,
) -> tuple[list[tuple[str, float, int, str]], dict]:
    """Return top-k straights for a member box using ONLY df_hist rows (already filtered to Date < play_date).

    Strategy:
      - If stream_key provided and enough samples in that stream for this box, use stream-specific counts.
      - Else fallback to global counts if enough global samples.
      - Else return [].

    Returns:
      (items, meta)
      items: [(straight, p, n, source)] sorted by p desc where p = count / n
      meta: dict with counts and which source used.
    """
    member_box4 = box_key(member_box4)
    if df_hist is None or df_hist.empty or not member_box4:
        return [], {"used": "none", "n": 0}

    if "Result" not in df_hist.columns:
        return [], {"used": "none", "n": 0}

    # compute box for each result (fast path: precompute once if missing)
    if "__BOXKEY__" not in df_hist.columns:
        tmp = df_hist.copy()
        tmp["__BOXKEY__"] = tmp["Result"].astype(str).map(box_key)
    else:
        tmp = df_hist

    def _counts(sub: pd.DataFrame):
        if sub is None or sub.empty:
            return pd.Series(dtype=int), 0
        vc = sub["Result"].astype(str).value_counts()
        n = int(vc.sum())
        return vc, n

    global_sub = tmp[tmp["__BOXKEY__"] == member_box4]
    global_vc, global_n = _counts(global_sub)

    used = "none"
    vc = global_vc
    n = global_n

    if stream_key:
        stream_sub = global_sub[global_sub["Stream"].astype(str) == str(stream_key)]
        stream_vc, stream_n = _counts(stream_sub)
        if stream_n >= int(min_stream_samples):
            used = "stream"
            vc = stream_vc
            n = stream_n
        elif global_n >= int(min_global_samples):
            used = "global"
        else:
            used = "none"
    else:
        used = "global" if global_n >= int(min_global_samples) else "none"

    if used == "none" or n <= 0:
        return [], {"used": used, "n": int(n), "global_n": int(global_n)}

    top = vc.head(int(top_k))
    items = []
    for straight, cnt in top.items():
        p = float(cnt) / float(n) if n else 0.0
        items.append((str(straight), p, int(n), used))
    return items, {"used": used, "n": int(n), "global_n": int(global_n), "stream_key": str(stream_key) if stream_key else None}


def _value_counts_result(df: pd.DataFrame) -> pd.Series:
    """Safe value_counts for Result column."""
    if df is None or df.empty:
        return pd.Series(dtype=int)
    if "Result" not in df.columns:
        return pd.Series(dtype=int)
    return df["Result"].astype(str).value_counts()

def _get_stream_result_counts(df_all: pd.DataFrame, stream: str) -> pd.Series:
    """Value counts of Result within a single stream."""
    if df_all is None or df_all.empty:
        return pd.Series(dtype=int)
    if "Stream" not in df_all.columns:
        return pd.Series(dtype=int)
    sub = df_all[df_all["Stream"].astype(str) == str(stream)]
    return _value_counts_result(sub)


def structure_of_4(d4: str) -> str:
    """Return AABC / AAAB / AABB / AAAA / ABCD based on counts."""
    from collections import Counter
    c = Counter(d4)
    counts = sorted(c.values(), reverse=True)
    if counts == [4]:
        return "AAAA"
    if counts == [3,1]:
        return "AAAB"
    if counts == [2,2]:
        return "AABB"
    if counts == [2,1,1]:
        return "AABC"
    return "ABCD"

def canonical_core_key(core: str) -> str:
    core = re.sub(r"\D", "", str(core))
    if len(core) == 3:
        return "".join(sorted(core))
    raise ValueError("Core must be 3 digits like 389")

def members_from_core(core_key: str, structure: str | None = None, **kwargs) -> List[str]:
    """Return 4-digit members for a 3-digit core.

    Two calling patterns are supported (backwards-compatible):
      1) members_from_core(core, "AABC") -> returns [AABC, ABBC, ABCC]
      2) members_from_core(core, include_family=True, include_aaab=True, include_aabb=True, include_aaaa=False)
         -> returns a combined, de-duplicated list of requested structures.
    """
    core_key = canonical_core_key(core_key)
    a, b, c = core_key[0], core_key[1], core_key[2]

    def _one(struct: str) -> List[str]:
        if struct == "AABC":
            x, y, z = f"{a}{a}{b}{c}", f"{a}{b}{b}{c}", f"{a}{b}{c}{c}"
        elif struct == "AAAB":
            x, y, z = f"{a}{a}{a}{b}", f"{a}{a}{a}{c}", f"{a}{b}{c}{c}"  # third is ABCC (rare engine uses it)
        elif struct == "AABB":
            x, y, z = f"{a}{a}{b}{b}", f"{a}{a}{c}{c}", f"{b}{b}{c}{c}"
        elif struct == "AAAA":
            x, y, z = f"{a}{a}{a}{a}", f"{b}{b}{b}{b}", f"{c}{c}{c}{c}"
        else:
            raise ValueError(f"Unknown structure: {struct}")
        return [box_key(x), box_key(y), box_key(z)]

    if structure is not None:
        return _one(structure)

    # Legacy / combined-call form
    include_family = bool(kwargs.get("include_family", True))
    include_aaab = bool(kwargs.get("include_aaab", False))
    include_aabb = bool(kwargs.get("include_aabb", False))
    include_aaaa = bool(kwargs.get("include_aaaa", False))

    out: List[str] = []
    if include_family:
        out.extend(_one("AABC"))
    if include_aaab:
        out.extend(_one("AAAB"))
    if include_aabb:
        out.extend(_one("AABB"))
    if include_aaaa:
        out.extend(_one("AAAA"))

    # De-duplicate while preserving order
    seen = set()
    res: List[str] = []
    for m in out:
        if m in seen:
            continue
        seen.add(m)
        res.append(m)
    return res



# ------------------------------------------------------------
# Core member labeling + member-pick prediction (walk-forward)
# ------------------------------------------------------------

@lru_cache(maxsize=4096)
def _core_member_label_map(core_key: str, include_rare: bool = False) -> dict[str, str]:
    """Map a core's member box-keys to human-readable member labels.

    Family (doubles) labels:
      - AABC = double of A (the first digit in sorted core)
      - ABBC = double of B
      - ABCC = double of C

    Rare labels (optional):
      - AAAB, AAAC (triple A with B/C)
      - AABB, AACC, BBCC
      - AAAA_A, AAAA_B, AAAA_C
    """
    core_key = canonical_core_key(core_key)
    a, b, c = core_key[0], core_key[1], core_key[2]

    m: dict[str, str] = {}

    # Family first (priority)
    fam_boxes = members_from_core(core_key, "AABC")
    for bk, lab in zip(fam_boxes, ["AABC", "ABBC", "ABCC"]):
        m.setdefault(str(bk), lab)

    if include_rare:
        # AAAB engine (note: third entry in members_from_core("AAAB") may overlap ABCC)
        aaab_boxes = members_from_core(core_key, "AAAB")
        for bk, lab in zip(aaab_boxes, ["AAAB", "AAAC", "ABCC"]):
            m.setdefault(str(bk), lab)

        aabb_boxes = members_from_core(core_key, "AABB")
        for bk, lab in zip(aabb_boxes, ["AABB", "AACC", "BBCC"]):
            m.setdefault(str(bk), lab)

        aaaa_boxes = members_from_core(core_key, "AAAA")
        for bk, lab in zip(aaaa_boxes, ["AAAA_A", "AAAA_B", "AAAA_C"]):
            m.setdefault(str(bk), lab)

    return m


def core_member_label(core_key: str, winner_4d: str, include_rare: bool = False) -> Optional[str]:
    """Return the member label for a core given a 4-digit winner (string).

    Uses box-key lookup first (fast and stable), then falls back to structure_of_4.
    """
    try:
        w = extract_4digit(winner_4d) or str(winner_4d).strip()
    except Exception:
        w = str(winner_4d).strip()
    if not w:
        return None
    bk = box_key(w)
    m = _core_member_label_map(core_key, include_rare=bool(include_rare))
    if bk in m:
        return m[bk]
    # Fallback (should be rare if winner is a member)
    try:
        return structure_of_4(w)
    except Exception:
        return None


def predict_core_member(
    df_all: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    *,
    basis: str = "core",
    stream: str | None = None,
    include_rare: bool = False,
) -> dict[str, Any]:
    """Predict which member label is most likely for this core (walk-forward safe).

    Prediction is based ONLY on rows with Date < test_date, restricted to the last `window_days`.

    basis:
      - "core": use all streams (global member distribution for that core)
      - "core_stream": use only that one stream (per-core-per-stream distribution)
    """
    if df_all is None or df_all.empty:
        return {"top1": None, "top2": None, "n": 0, "counts": {}}

    # Window slice: [test_date - window_days, test_date)
    try:
        td = pd.to_datetime(test_date).normalize()
    except Exception:
        td = pd.Timestamp(test_date).normalize()
    start = td - pd.Timedelta(days=int(window_days))

    sub = df_all
    try:
        if "Date" in sub.columns:
            sub = sub[sub["Date"].notna()]
            sub = sub[(sub["Date"] >= start) & (sub["Date"] < td)]
    except Exception:
        pass

    if basis == "core_stream" and stream is not None and "Stream" in sub.columns:
        try:
            sub = sub[sub["Stream"].astype(str) == str(stream)]
        except Exception:
            pass

    label_map = _core_member_label_map(core_key, include_rare=bool(include_rare))
    member_boxes = set(label_map.keys())
    box_col = "BoxKey4" if "BoxKey4" in sub.columns else ("Box" if "Box" in sub.columns else None)

    if box_col is None:
        # As a last resort, compute box keys on the fly
        try:
            tmp = sub.copy()
            tmp["_bk4"] = tmp["Result"].astype(str).map(box_key)
            box_col = "_bk4"
            sub = tmp
        except Exception:
            return {"top1": None, "top2": None, "n": 0, "counts": {}}

    try:
        hit_rows = sub[sub[box_col].astype(str).isin(member_boxes)]
    except Exception:
        hit_rows = pd.DataFrame()

    if hit_rows is None or hit_rows.empty:
        return {"top1": None, "top2": None, "n": 0, "counts": {}}

    # Map box keys -> labels (fast), then count
    labs = hit_rows[box_col].astype(str).map(label_map)
    counts = labs.value_counts(dropna=True)
    if counts.empty:
        return {"top1": None, "top2": None, "n": 0, "counts": {}}

    top = counts.index.tolist()
    top1 = top[0] if len(top) >= 1 else None
    top2 = top[1] if len(top) >= 2 else None
    return {
        "top1": top1,
        "top2": top2,
        "n": int(counts.sum()),
        "counts": counts.to_dict(),
    }



def _member_last_label(
    df_all: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    *,
    stream: str | None = None,
) -> tuple[Optional[str], int]:
    """Return LAST observed family-member label (AABC/ABBC/ABCC) for a core in the lookback window.
    Returns (label, n_hits_in_window). Walk-forward safe (uses Date < test_date only).
    """
    if df_all is None or df_all.empty:
        return (None, 0)
    td = pd.to_datetime(test_date).normalize()
    start = td - pd.Timedelta(days=int(window_days))
    sub = df_all
    if "Date" in sub.columns:
        sub = sub[sub["Date"].notna()]
        sub = sub[(sub["Date"] >= start) & (sub["Date"] < td)]
    if stream is not None and "Stream" in sub.columns:
        sub = sub[sub["Stream"].astype(str) == str(stream)]
    label_map = _core_member_label_map(core_key, include_rare=False)
    member_boxes = set(label_map.keys())
    box_col = "BoxKey4" if "BoxKey4" in sub.columns else ("Box" if "Box" in sub.columns else None)
    if box_col is None:
        tmp = sub.copy()
        tmp["_bk4"] = tmp["Result"].astype(str).map(box_key)
        box_col = "_bk4"
        sub = tmp
    hit_rows = sub[sub[box_col].astype(str).isin(member_boxes)].copy()
    if hit_rows.empty:
        return (None, 0)
    hit_rows = hit_rows.sort_values("Date")
    last_bk = hit_rows.iloc[-1][box_col]
    lab = label_map.get(str(last_bk))
    return (lab if lab in ("AABC", "ABBC", "ABCC") else None, int(len(hit_rows)))


def _seed_for_stream_asof(df_all: pd.DataFrame, stream: str, asof_date: pd.Timestamp) -> Optional[str]:
    """Most recent 4-digit result for a stream strictly before asof_date."""
    if df_all is None or df_all.empty:
        return None
    td = pd.to_datetime(asof_date).normalize()
    sub = df_all[(df_all["Date"] < td) & (df_all["Stream"].astype(str) == str(stream))].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("Date")
    return str(sub.iloc[-1].get("Result", "")).strip() or None


def _seed_traits_for_core_stream(
    df_all: pd.DataFrame,
    core_key: str,
    stream: str,
    asof_date: pd.Timestamp,
) -> dict[str, str]:
    """Compute the standard seed-trait fields used in the seed-traits CSVs, for rulecards."""
    seed = _seed_for_stream_asof(df_all, stream, asof_date)
    if not seed:
        return {}
    seed = extract_4digit(seed) or seed
    digs = [int(ch) for ch in str(seed).zfill(4) if ch.isdigit()]
    if len(digs) != 4:
        return {}
    core_digs = set(str(core_key).zfill(3))
    ssum = sum(digs)
    spread = max(digs) - min(digs)
    even_cnt = sum(1 for d in digs if d % 2 == 0)
    high_cnt = sum(1 for d in digs if d >= 5)
    traits = {
        "seed_structure": structure_of_4(str(seed).zfill(4)),
        "seed_spread": ("<=2" if spread <= 2 else ("3-5" if spread <= 5 else ">=6")),
        "seed_even_count": str(even_cnt),
        "seed_high_count": str(high_cnt),
        "seed_sum_mod2": str(ssum % 2),
        "seed_sum_mod3": str(ssum % 3),
        # Sliding 4-sum band: (sum-1) to (sum+2) matches labels like 3-6, 11-14, etc.
        "seed_sum_range4_best": f"{ssum-1}-{ssum+2}",
        "seed_first_in_core": ("yes" if str(seed).zfill(4)[0] in core_digs else "no"),
        "seed_last_in_core": ("yes" if str(seed).zfill(4)[-1] in core_digs else "no"),
        "overlap_unique": str(len(set(str(seed).zfill(4)) & core_digs)),
        "seed_contains_core_pair": ("yes" if len(set(str(seed).zfill(4)) & core_digs) >= 2 else "no"),
    }
    # grid_last5_core_digits: how many of the core digits appear in the last-5 union digits for this stream (as of asof_date)
    try:
        sub = df_all[(df_all["Stream"].astype(str) == str(stream)) & (df_all["Date"] < pd.to_datetime(asof_date).normalize())].copy()
        sub = sub.sort_values("Date").tail(5)
        union = set("".join(sub["Result"].astype(str).tolist()))
        cnt = len(core_digs & union)
        if cnt == 3:
            traits["grid_last5_core_digits"] = "3"
        elif cnt >= 2:
            traits["grid_last5_core_digits"] = ">=2"
        else:
            traits["grid_last5_core_digits"] = str(cnt)
    except Exception:
        pass
    return traits


def _pick_best_seed_trait_rule(traits_pos_df: pd.DataFrame, core_key: str) -> Optional[tuple[str, str]]:
    """From the positive seed-traits CSV, pick the single highest-lift (trait, value) for this core."""
    if traits_pos_df is None or traits_pos_df.empty:
        return None
    ck = int(str(core_key).zfill(3))
    sub = traits_pos_df.copy()
    # core_family column in these CSVs is numeric (e.g., 12 for 012)
    sub = sub[sub["core_family"].astype(int) == ck]
    if sub.empty:
        return None
    sub = sub.sort_values(["lift", "trait_hits"], ascending=[False, False])
    r = sub.iloc[0]
    return (str(_safe_get(r,"trait","")).strip(), str(_safe_get(r,"value","")).strip())


def _member_mode_from_trait(
    df_all: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    trait_name: str,
    trait_value: str,
    *,
    stream: str | None = None,
) -> Optional[str]:
    """Within the walk-forward window, among hits where (trait==value) at the seed, return the MODE member label."""
    if not trait_name or trait_value is None:
        return None
    td = pd.to_datetime(test_date).normalize()
    start = td - pd.Timedelta(days=int(window_days))
    # Build stream-day transitions: seed = last result before day, winner = day's result.
    sub = df_all
    if "Date" not in sub.columns:
        return None
    sub = sub[sub["Date"].notna()].copy()
    sub = sub[(sub["Date"] >= start) & (sub["Date"] < td)]
    if stream is not None:
        sub = sub[sub["Stream"].astype(str) == str(stream)]
    if sub.empty:
        return None
    label_map = _core_member_label_map(core_key, include_rare=False)
    member_boxes = set(label_map.keys())
    # Keep only rows where the RESULT is a member of this core
    box_col = "BoxKey4" if "BoxKey4" in sub.columns else ("Box" if "Box" in sub.columns else None)
    if box_col is None:
        sub = sub.copy()
        sub["_bk4"] = sub["Result"].astype(str).map(box_key)
        box_col = "_bk4"
    hit = sub[sub[box_col].astype(str).isin(member_boxes)].copy()
    if hit.empty:
        return None
    # Compute trait per row based on the *seed* for that stream at that date (as-of that date)
    vals = []
    for _, r in hit.iterrows():
        s = str(_safe_get(r,"Stream",""))
        d = pd.to_datetime(_safe_get(r,"Date")).normalize()
        t = _seed_traits_for_core_stream(df_all, core_key, s, d).get(trait_name)
        vals.append(t)
    hit["_trait_val"] = vals
    hit = hit[hit["_trait_val"].astype(str) == str(trait_value)]
    if hit.empty:
        return None
    labs = hit[box_col].astype(str).map(label_map)
    vc = labs.value_counts()
    if vc.empty:
        return None
    top = vc.index.tolist()[0]
    return top if top in ("AABC","ABBC","ABCC") else None


def _member_prediction_variants(
    df_all: pd.DataFrame,
    traits_pos_df: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    *,
    stream: str,
    basis: str,
    min_stream_hits_for_last: int = 3,
    enable_member_seed_traits_overrides: bool = False,
    member_seed_traits_weight: float = 0.25,
    member_traits_pos_lut: Optional[Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = None,
    member_traits_neg_lut: Optional[Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = None,
) -> dict[str, Optional[str]]:
    """Compute member Top1 predictions under multiple strategies, walk-forward safe."""
    # MODE (same as existing predictor)
    mp_mode = predict_core_member(df_all, core_key, test_date, window_days, basis=("core_stream" if basis=="core_stream" else "core"), stream=(stream if basis=="core_stream" else None), include_rare=False)
    pred_mode = mp_mode.get("top1")
    # LAST(global)
    last_g, _ = _member_last_label(df_all, core_key, test_date, window_days, stream=None)
    # LAST(stream)
    last_s, n_s = _member_last_label(df_all, core_key, test_date, window_days, stream=stream)
    # Hierarchical LAST: use LAST(stream) if enough samples, else LAST(global), else MODE
    pred_last_h = None
    if last_s is not None and n_s >= int(min_stream_hits_for_last):
        pred_last_h = last_s
    elif last_g is not None:
        pred_last_h = last_g
    else:
        pred_last_h = pred_mode
    # Seed-structure override
    pred_seed_ovr = pred_last_h
    try:
        seed = _seed_for_stream_asof(df_all, stream, test_date)
        if seed:
            sstruct = structure_of_4(extract_4digit(seed) or str(seed).zfill(4))
            override = {"AAAB": "AABC", "AABB": "ABBC", "AAAA": "ABCC"}.get(str(sstruct))
            if override in ("AABC","ABBC","ABCC"):
                pred_seed_ovr = override
    except Exception:
        pass
    # Trait-lift override: use best (trait,value) for this core, then MODE among past hits where that trait fires
    pred_trait_ovr = pred_last_h
    best = _pick_best_seed_trait_rule(traits_pos_df, core_key)
    if best is not None:
        tname, tval = best
        # if the current seed's trait matches, apply the member-mode for that trait value
        cur_traits = _seed_traits_for_core_stream(df_all, core_key, stream, test_date)
        if cur_traits.get(tname) == tval:
            m = _member_mode_from_trait(df_all, core_key, test_date, window_days, tname, tval, stream=(stream if basis=="core_stream" else None))
            if m in ("AABC","ABBC","ABCC"):
                pred_trait_ovr = m

    # Optional: member-level seed-trait soft override (uses uploaded member traits tables)
    pred_member_trait_ovr: Optional[str] = None
    pred_member_trait_top2: Optional[str] = None
    if enable_member_seed_traits_overrides and (member_traits_pos_lut or member_traits_neg_lut):
        try:
            # base probabilities from MODE counts (fallback to uniform)
            counts = mp_mode.get("counts", {}) if isinstance(mp_mode, dict) else {}
            total = float(sum(float(v) for v in counts.values())) if counts else 0.0
            if total > 0:
                base_probs = {k: float(v) / total for k, v in counts.items()}
            else:
                base_probs = {k: 1.0 for k in ("AABC","ABBC","ABCC")}

            seed = _seed_for_stream_asof(df_all, stream, test_date)
            top1, top2 = _apply_member_traits_soft_override(
                base_probs=base_probs,
                core=core_key,
                seed=seed,
                stream=str(stream),
                test_date=test_date.date() if hasattr(test_date, "date") else test_date,
                df_all=df_all,
                member_pos_lut=(member_traits_pos_lut or {}),
                member_neg_lut=(member_traits_neg_lut or {}),
                weight=float(member_seed_traits_weight or 0.25),
            )
            pred_member_trait_ovr = top1
            pred_member_trait_top2 = top2[1] if len(top2) > 1 else None
        except Exception:
            pred_member_trait_ovr = None
            pred_member_trait_top2 = None

    return {
        "MODE": pred_mode,
        "LAST_GLOBAL": last_g,
        "LAST_HIER": pred_last_h,
        "SEED_OVERRIDE": pred_seed_ovr,
        "TRAIT_OVERRIDE": pred_trait_ovr,
        "MEMBER_TRAIT_OVERRIDE": pred_member_trait_ovr,
        "MEMBER_TRAIT_TOP2": pred_member_trait_top2,
    }

def try_read_tablelike(uploaded) -> pd.DataFrame:
    """
    Accept .csv or LotteryPost tab .txt.
    Expected columns (any case): Date, State, Game, Results (or Result/Winning Numbers).
    """
    if uploaded is None:
        return pd.DataFrame()

    name = getattr(uploaded, "name", "") or ""
    # try csv first
    try:
        df = pd.read_csv(uploaded)
        if df.shape[1] == 1:
            raise ValueError("Looks like 1-column; try tab.")
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep="\t", header=None)
        # try to name columns if 4+ cols
        if df.shape[1] >= 4:
            df = df.iloc[:, :4]
            df.columns = ["Date", "State", "Game", "Results"]
        else:
            # fallback
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # normalize column names
    colmap = {c.lower().strip(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in colmap:
                return colmap[c]
        return None

    date_col = pick("date")
    state_col = pick("state")
    game_col = pick("game")
    results_col = pick("results", "result", "winning numbers", "winning_numbers", "winningnumbers")

    if date_col is None or state_col is None or game_col is None or results_col is None:
        # best-effort: if there are exactly 4 columns, assume those
        if df.shape[1] >= 4:
            df = df.iloc[:, :4].copy()
            df.columns = ["Date", "State", "Game", "Results"]
        else:
            raise ValueError("Could not detect Date/State/Game/Results columns.")
    else:
        df = df.rename(columns={
            date_col: "Date",
            state_col: "State",
            game_col: "Game",
            results_col: "Results",
        })

    # parse date
    df["Date"] = _parse_date_series(df["Date"])
    df = df[df["Date"].notna()].copy()

    # parse results
    df["Pick4"] = df["Results"].map(extract_pick4_digits)
    df = df[df["Pick4"].notna()].copy()
    # compatibility aliases used in other modules
    df["Result"] = df["Pick4"]

    df["Structure"] = df["Pick4"].map(structure_of_4)
    df["Box"] = df["Pick4"].map(box_key)
    df["BoxKey4"] = df["Box"]
    df["Stream"] = df["State"].astype(str).str.strip() + " | " + df["Game"].astype(str).str.strip()
    return df



def try_read_picklist(uploaded) -> pd.DataFrame:
    """
    Accept a simple list of Pick4 numbers (previous-day file) in .txt or .csv form.
    Extracts 4-digit sequences anywhere in the file.
    Returns a dataframe with: Result, Pick4, Box, BoxKey4, Structure.
    """
    if uploaded is None:
        return pd.DataFrame()

    try:
        raw = uploaded.read()
    except Exception:
        raw = None

    # Reset pointer for possible re-reads by caller
    try:
        uploaded.seek(0)
    except Exception:
        pass

    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, bytes):
        try:
            s = raw.decode("utf-8", errors="ignore")
        except Exception:
            s = str(raw)
    else:
        s = str(raw)

    # Common case: one number per line, but we accept any separators
    nums = re.findall(r"(?<!\d)(\d{4})(?!\d)", s)
    if not nums:
        # try to read as a one-column csv
        try:
            uploaded.seek(0)
        except Exception:
            pass
        try:
            df1 = pd.read_csv(uploaded, header=None)
            flat = []
            for v in df1.iloc[:, 0].astype(str).tolist():
                flat += re.findall(r"(?<!\d)(\d{4})(?!\d)", v)
            nums = flat
        except Exception:
            nums = []

    nums = [str(n).zfill(4) for n in nums if n is not None]
    # de-dupe while preserving order
    seen = set()
    out = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)

    df = pd.DataFrame({"Result": out})
    if df.empty:
        return df
    df["Pick4"] = df["Result"]
    df["Structure"] = df["Pick4"].map(structure_of_4)
    df["Box"] = df["Pick4"].map(box_key)
    df["BoxKey4"] = df["Box"]
    return df


# -------------------------
# Stats + ranking
# -------------------------

@dataclass
class RankConfig:
    # History window used for rank stats (default 180, switchable to 365 in UI)
    window_days: int = 180

    # Bucket method:
    # - Top 'top_base' by BaseScore (HitsPerWeek)
    # - From base ranks due_from_rank..due_to_rank, take Top 'top_due' by DueIndex (DaysSinceLastHit)
    top_base: int = 12
    due_from_rank: int = 13
    due_to_rank: int = 60
    top_due: int = 8

    # Display / scoring knobs (kept as soft signals)
    max_master_rows: int = 120
    max_final_rows: int = 300
    include_24h_signals: bool = True
    pos_strength_weight: float = 0.25
    seed_core_key: str = "core"  # reserved for future compatibility

    # Back-compat aliases (older UI keys)
    @property
    def top12(self) -> int:
        return int(self.top_base)

    @property
    def due_ranks(self):
        return (int(self.due_from_rank), int(self.due_to_rank))

@property
def top_n(self) -> int:
    """Legacy alias: some older builds referenced RankConfig.top_n."""
    return int(self.top_base)

@property
def base_top_n(self) -> int:
    """Legacy alias for the Top bucket size."""
    return int(self.top_base)

@property
def due_top_n(self) -> int:
    """Legacy alias for the Due bucket size."""
    return int(self.top_due)


def within_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    max_date = df["Date"].max()
    cutoff = max_date - pd.Timedelta(days=days)
    return df[df["Date"] >= cutoff].copy()

def compute_core_hits(df: pd.DataFrame, core: str, structures: Iterable[str]) -> pd.DataFrame:
    """
    Return df subset containing only rows that are hits for the core, for the chosen structures.
    We match by Box membership, so order doesn't matter.
    """
    core = canonical_core_key(core)
    boxes = set()
    for s in structures:
        for mem in members_from_core(core, s):
            boxes.add(box_key(mem))
    return df[df["Box"].isin(boxes)].copy()

def stream_summary(df_all: pd.DataFrame, df_hits: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    For each stream, compute:
    - draws_window, hits_window, hits_per_week_window
    - days_since_last_hit_window (based on df_hits within full history)
    """
    if df_all.empty:
        return pd.DataFrame(columns=[
            "Stream","DrawsWindow","HitsWindow","HitsPerWeek","LastHitDate","DaysSinceLastHit"
        ])

    dfw = within_last_days(df_all, window_days)
    max_date = df_all["Date"].max()

    draws = dfw.groupby("Stream").size().rename("DrawsWindow")
    hitsw = within_last_days(df_hits, window_days).groupby("Stream").size().rename("HitsWindow")

    # last hit date from full history (not just window) for "due"
    last_hit = df_hits.groupby("Stream")["Date"].max().rename("LastHitDate")

    out = pd.concat([draws, hitsw, last_hit], axis=1).fillna({"HitsWindow":0})
    out["HitsWindow"] = out["HitsWindow"].astype(int)
    out["DrawsWindow"] = out["DrawsWindow"].astype(int)

    weeks = max(window_days / 7.0, 1e-9)
    out["HitsPerWeek"] = out["HitsWindow"] / weeks

    out["DaysSinceLastHit"] = (max_date - out["LastHitDate"]).dt.days
    out.loc[out["LastHitDate"].isna(), "DaysSinceLastHit"] = 0

    out = out.reset_index().sort_values(["HitsPerWeek","HitsWindow"], ascending=False)
    out["RankPos"] = np.arange(1, len(out)+1)
    
    # Derived ranking columns (for bucket picks + backtest)
    # BaseScoreRank: same as RankPos (1 = strongest recent strength)
    if "RankPos" in out.columns and "BaseScoreRank" not in out.columns:
        out["BaseScoreRank"] = out["RankPos"]
    # BaseScore: a simple continuous strength proxy (used for sorting/UX)
    if "BaseScore" not in out.columns:
        out["BaseScore"] = out.get("HitsPerWeek", 0.0)
    # DueIndex: "how due" a stream is (days since last hit)
    if "DueIndex" not in out.columns:
        out["DueIndex"] = out.get("DaysSinceLastHit", 0)
    # DueIndexRank: 1 = most due (largest DueIndex)
    if "DueIndexRank" not in out.columns:
        try:
            _di = pd.to_numeric(out["DueIndex"], errors="coerce").fillna(-1)
            out["DueIndexRank"] = (-_di).rank(method="dense", ascending=True).astype(int)
        except Exception:
            out["DueIndexRank"] = out["BaseScoreRank"] if "BaseScoreRank" in out.columns else range(1, len(out) + 1)

    return out

def position_percentile_map(df_rankpos: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a percentile map over RankPos (1..78) using hit counts.

    Input must have at least:
      - RankPos (int)
      - HitsWindow (int/float) OR HitCount (int/float)

    Returns:
      (pos_df, meta)

    pos_df columns include:
      - RankPos
      - HitCount
      - HitCountPctile (0-100)
      - PctStrength (alias of HitCountPctile; back-compat)
      - HitSharePct (percent of total hits at this RankPos)
      - CumuHitSharePct (cumulative share across RankPos ascending)

    Notes:
      - If multiple rows share the same RankPos (e.g., aggregating many cores),
        they are summed first.
    """
    empty_cols = [
        "RankPos", "HitCount", "HitCountPctile", "PctStrength", "HitSharePct", "CumuHitSharePct",
        "HitShare", "CumHitShare",
    ]
    if df_rankpos is None or df_rankpos.empty:
        return pd.DataFrame(columns=empty_cols), {"total_hits": 0.0, "rows": 0}

    pos = df_rankpos.copy()

    # Normalize column name
    if "HitCount" not in pos.columns and "HitsWindow" in pos.columns:
        pos = pos.rename(columns={"HitsWindow": "HitCount"})
    if "HitCount" not in pos.columns:
        # Best effort: try common alternatives
        for alt in ["Hits", "Count", "Hit_Count"]:
            if alt in pos.columns:
                pos = pos.rename(columns={alt: "HitCount"})
                break

    if "RankPos" not in pos.columns or "HitCount" not in pos.columns:
        return pd.DataFrame(columns=empty_cols), {"total_hits": 0.0, "rows": 0}

    # Aggregate by RankPos (important for ALL-CORES maps)
    pos = pos.groupby("RankPos", as_index=False)["HitCount"].sum()

    # Sort by RankPos for consistent cumulative share
    pos["RankPos"] = pos["RankPos"].astype(int)
    pos = pos.sort_values("RankPos").reset_index(drop=True)

    # Percentile rank by hit count (ties handled by average rank)
    pos["HitCountPctile"] = pos["HitCount"].rank(pct=True) * 100.0

    total_hits = float(pos["HitCount"].sum())
    denom = total_hits if total_hits != 0.0 else 1.0
    pos["HitSharePct"] = (pos["HitCount"] / denom) * 100.0
    pos["CumuHitSharePct"] = pos["HitSharePct"].cumsum()

    # Back-compat aliases used in older UI text + newer tie-break key
    pos["HitShare"] = pos["HitSharePct"]
    pos["CumHitShare"] = pos["CumuHitSharePct"]
    pos["PctStrength"] = pos["HitCountPctile"]

    meta = {
        "total_hits": total_hits,
        "rows": int(pos.shape[0]),
    }
    return pos, meta


# -------------------------
# Seed Traits (positive/negative) + Cadence (v51)
# -------------------------
def _read_local_or_uploaded_csv(uploaded_file, local_path: str) -> pd.DataFrame:
    """Read CSV from uploaded file-like or a local repo path. Returns empty df on failure."""
    try:
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            return pd.read_csv(uploaded_file)
    except Exception:
        pass
    try:
        if local_path and os.path.exists(local_path):
            return pd.read_csv(local_path)
    except Exception:
        pass
    return pd.DataFrame()

def _read_local_or_uploaded_text(uploaded_file, local_path: str) -> str:
    try:
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            return uploaded_file.read().decode('utf-8', errors='ignore') if hasattr(uploaded_file, 'read') else str(uploaded_file)
    except Exception:
        pass
    try:
        if local_path and os.path.exists(local_path):
            return open(local_path, 'r', encoding='utf-8', errors='ignore').read()
    except Exception:
        pass
    return ""




def _parse_cadence_table_from_text(text: str) -> pd.DataFrame | None:
    """Parse cadence export text.

    Supports:
      1) CSV/TSV exports (preferred) with headers like:
         level,core_family,member,stream,hits_all,mean_gap_days,days_since_last_hit,...
      2) Markdown-like tables (| separated).

    IMPORTANT: Streams can contain the literal pipe character (e.g. 'ALL | ALL'),
    so we MUST try CSV/TSV parsing FIRST before treating '|' as a column delimiter.
    """
    if text is None:
        return None

    raw = str(text).lstrip("\ufeff")
    raw_lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not raw_lines:
        return None

    header = raw_lines[0].strip().lower()

    # --- 1) CSV / TSV fast-path (robust to pipes inside stream names) ---
    # Detect delimiter by header signature.
    if ("level" in header and "core_family" in header and "stream" in header) and ("," in header or "\t" in header or ";" in header):
        import io as _io
        # Try common delimiters; keep everything as strings first, then coerce numerics.
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(_io.StringIO(raw), sep=sep, engine="python")
                df.columns = [str(c).strip().lower() for c in df.columns]
                if {"level", "core_family", "stream"}.issubset(set(df.columns)):
                    # normalize expected columns (lower -> app's canonical names)
                    # Fill missing optional columns with NaN
                    want_cols = {
                        "level": "level",
                        "core_family": "core_family",
                        "member": "member",
                        "stream": "stream",
                        "hits_all": "hits_all",
                        "hits_per_week": "hits_per_week",
                        "median_gap_days": "median_gap_days",
                        "avg_gap_days": "mean_gap_days",  # accept avg_gap_days as mean_gap_days
                        "mean_gap_days": "mean_gap_days",
                        "days_since_last_hit": "days_since_last_hit",
                    }
                    # rename with mapping when present
                    ren = {}
                    for c in df.columns:
                        if c in want_cols:
                            ren[c] = want_cols[c]
                    df = df.rename(columns=ren)

                    # Ensure required columns exist
                    if "member" not in df.columns:
                        df["member"] = ""
                    # Coerce key cols to strings
                    df["level"] = df["level"].fillna("").astype(str).str.strip().str.lower()
                    df["core_family"] = df["core_family"].fillna("").astype(str).str.strip()
                    df["member"] = df["member"].fillna("").astype(str).str.strip()
                    df["stream"] = df["stream"].fillna("").astype(str).str.strip()

                    # Numeric coercions (safe)
                    for num_col in ["hits_all", "hits_per_week", "median_gap_days", "mean_gap_days", "days_since_last_hit"]:
                        if num_col in df.columns:
                            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

                    df["core_key"] = df["core_family"].astype(str).str.strip().str.lower()
                    df["stream_key"] = df["stream"].astype(str).str.strip().str.lower()
                    df["member_key"] = df["member"].astype(str).str.strip().str.lower()

                    # Canonical level names
                    df.loc[df["level"].isin(["stream" , "streams"]), "level"] = "stream"
                    df.loc[df["level"].isin(["core", "cores", "family", "families"]), "level"] = "core"
                    df.loc[df["level"].isin(["member", "members"]), "level"] = "member"

                    return df
            except Exception:
                continue
        # fall through to markdown parsing if csv attempts fail

    # --- 2) Markdown-ish pipe table fallback ---
    lines = raw_lines

    # Find a likely header line (pipe-delimited)
    header_idx = None
    for i, ln in enumerate(lines[:20]):  # only scan first chunk for header
        if "|" in ln and ("level" in ln.lower()) and ("core" in ln.lower()) and ("stream" in ln.lower()):
            header_idx = i
            break
    if header_idx is None:
        return None

    # Collect table lines from header onward while pipes appear
    table_lines = []
    for ln in lines[header_idx:]:
        if "|" in ln:
            table_lines.append(ln)
        else:
            # stop once table ends
            if table_lines:
                break

    if len(table_lines) < 2:
        return None

    # Split pipe table
    rows = []
    for ln in table_lines:
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        rows.append(parts)

    # Drop separator rows like '---'
    rows = [r for r in rows if not all(re.fullmatch(r"-+", (c or "").strip()) for c in r)]

    if not rows or len(rows) < 2:
        return None

    headers = [h.strip().lower() for h in rows[0]]
    data = rows[1:]

    # Build DataFrame with best-effort column alignment
    max_len = max(len(headers), max(len(r) for r in data))
    headers = (headers + [f"col_{i}" for i in range(len(headers), max_len)])[:max_len]
    norm_data = [(r + [""] * (max_len - len(r)))[:max_len] for r in data]

    df = pd.DataFrame(norm_data, columns=headers)

    # Normalize known columns
    colmap = {
        "avg_gap_days": "mean_gap_days",
        "mean_gap_days": "mean_gap_days",
        "days_since_last_hit": "days_since_last_hit",
        "hits_all": "hits_all",
        "hits_per_week": "hits_per_week",
        "median_gap_days": "median_gap_days",
    }
    df = df.rename(columns=colmap)

    # Ensure expected columns exist
    for c in ["level", "core_family", "member", "stream"]:
        if c not in df.columns:
            df[c] = ""

    df["level"] = df["level"].fillna("").astype(str).str.strip().str.lower()
    df["core_family"] = df["core_family"].fillna("").astype(str).str.strip()
    df["member"] = df["member"].fillna("").astype(str).str.strip()
    df["stream"] = df["stream"].fillna("").astype(str).str.strip()

    for num_col in ["hits_all", "hits_per_week", "median_gap_days", "mean_gap_days", "days_since_last_hit"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df["core_key"] = df["core_family"].astype(str).str.strip().str.lower()
    df["stream_key"] = df["stream"].astype(str).str.strip().str.lower()
    df["member_key"] = df["member"].astype(str).str.strip().str.lower()

    df.loc[df["level"].isin(["stream" , "streams"]), "level"] = "stream"
    df.loc[df["level"].isin(["core", "cores", "family", "families"]), "level"] = "core"
    df.loc[df["level"].isin(["member", "members"]), "level"] = "member"

    return df
def _build_traits_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """lookup[core_family][trait][value] -> lift"""
    lookup: Dict[str, Dict[str, Dict[str, float]]] = {}
    if df is None or df.empty:
        return lookup
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    core_col = cols.get("core_family", None)
    trait_col = cols.get("trait", None)
    val_col = cols.get("value", None)
    lift_col = cols.get("lift", None)
    if not (core_col and trait_col and val_col and lift_col):
        return lookup
    for _, r in df.iterrows():
        try:
            core_raw = str(r[core_col]).strip()
            # Normalize to 3-digit core string (preserve leading zeros)
            core = core_raw.zfill(3) if core_raw.isdigit() else core_raw
        except Exception:
            continue
        trait = str(r[trait_col]).strip()
        val = str(r[val_col]).strip()
        try:
            lift = float(r[lift_col])
        except Exception:
            lift = 1.0
        if not trait:
            continue
        lookup.setdefault(core, {}).setdefault(trait, {})[val] = lift
    return lookup


def _member_label_from_core_and_member(core: str, mem4: str) -> Optional[str]:
    """
    Derive member label (AABC/ABBC/ABCC) for a 3-digit core and a 4-digit member value.

    IMPORTANT: member integers in your exported tables are stored in a canonical form
    (duplicate digit repeated first), so the raw 4-digit pattern (e.g., 1109) does NOT
    correspond to the family label by positional pattern. We must infer which core digit
    is duplicated.
    """
    core = str(core or "").strip()
    core = core.zfill(3) if core.isdigit() else core
    mem4 = str(mem4 or "").strip()
    mem4 = extract_4digit(mem4) or mem4
    if mem4.isdigit():
        mem4 = mem4.zfill(4)
    if not (core.isdigit() and mem4.isdigit() and len(core) == 3 and len(mem4) == 4):
        return None

    # Core digits A,B,C are the three distinct digits in the core (ascending)
    core_digits = [int(ch) for ch in core]
    uniq = sorted(set(core_digits))
    if len(uniq) != 3:
        return None
    A, B, C = uniq[0], uniq[1], uniq[2]

    # Count digits in member
    digs = [int(ch) for ch in mem4]
    from collections import Counter
    cnt = Counter(digs)
    dup = None
    for d, n in cnt.items():
        if n == 2:
            dup = d
            break
    if dup is None:
        return None

    if dup == A:
        return "AABC"
    if dup == B:
        return "ABBC"
    if dup == C:
        return "ABCC"
    return None


def _build_member_traits_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """lookup[core_family][member_label][trait][value] -> lift

    member_label is one of AABC/ABBC/ABCC inferred from which core digit is duplicated.
    This is required because member numbers are stored in a canonical order where the
    duplicate digit is placed first (e.g., core 019 member 1109 corresponds to ABBC).
    """
    lookup: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    if df is None or df.empty:
        return lookup
    cols = {c.lower(): c for c in df.columns}
    core_col = cols.get("core_family", None)
    member_col = cols.get("member", None)
    trait_col = cols.get("trait", None)
    val_col = cols.get("value", None)
    lift_col = cols.get("lift", None)
    if not (core_col and member_col and trait_col and val_col and lift_col):
        return lookup

    for _, r in df.iterrows():
        try:
            core_raw = str(r[core_col]).strip()
            core = core_raw.zfill(3) if core_raw.isdigit() else core_raw
            mem_raw = str(r[member_col]).strip()
            mem = mem_raw.zfill(4) if mem_raw.isdigit() else mem_raw
            mem = extract_4digit(mem) or mem

            mem_label = _member_label_from_core_and_member(core, mem)
            if mem_label not in ("AABC", "ABBC", "ABCC"):
                continue

            trait = str(r[trait_col]).strip()
            value = str(r[val_col]).strip()
            lift = float(r[lift_col])
            if not trait:
                continue
            lookup.setdefault(core, {}).setdefault(mem_label, {}).setdefault(trait, {})[value] = lift
        except Exception:
            continue
    return lookup

    cols = {c.lower(): c for c in df.columns}
    core_col = cols.get("core_family", None)
    member_col = cols.get("member", None)
    trait_col = cols.get("trait", None)
    val_col = cols.get("value", None)
    lift_col = cols.get("lift", None)
    if not (core_col and member_col and trait_col and val_col and lift_col):
        return lookup

    for _, r in df.iterrows():
        try:
            core_raw = str(r[core_col]).strip()
            core = core_raw.zfill(3) if core_raw.isdigit() else core_raw
            mem_raw = str(r[member_col]).strip()
            mem = mem_raw.zfill(4) if mem_raw.isdigit() else mem_raw
            mem = extract_4digit(mem) or mem
            mem_label = structure_of_4(mem)
            if mem_label not in ("AABC","ABBC","ABCC"):
                continue
            trait = str(r[trait_col]).strip()
            value = str(r[val_col]).strip()
            lift = float(r[lift_col])
            lookup.setdefault(core, {}).setdefault(mem_label, {}).setdefault(trait, {})[value] = lift
        except Exception:
            continue
    return lookup


def _seed_traits_extended(seed: str) -> Dict[str, str]:
    """Extra seed traits used by member-trait tables (kept separate to avoid regressions)."""
    seed = str(seed or "").strip()
    seed = extract_4digit(seed) or seed
    if seed.isdigit():
        seed = seed.zfill(4)
    digs = [int(ch) for ch in seed if ch.isdigit()]
    if len(digs) != 4:
        return {}
    s = sum(digs)
    spread = max(digs) - min(digs)

    # lead/trail bands
    def _band(d: int) -> str:
        if d <= 3:
            return "0-3"
        if d <= 6:
            return "4-6"
        return "7-9"

    lead_band = _band(digs[0])
    trail_band = _band(digs[-1])

    # spread band (simple, consistent buckets)
    if spread <= 3:
        spread_band = "0-3"
    elif spread <= 6:
        spread_band = "4-6"
    else:
        spread_band = "7-9"

    # sum band (coarse)
    if s <= 9:
        sum_band = "0-9"
    elif s <= 19:
        sum_band = "10-19"
    elif s <= 29:
        sum_band = "20-29"
    else:
        sum_band = "30-36"

    # consecutive pairs (positional)
    consec = 0
    for a, b in zip(digs, digs[1:]):
        if abs(a - b) == 1:
            consec += 1

    prime_set = {2, 3, 5, 7}
    prime_count = sum(1 for d in digs if d in prime_set)

    # VTRAC groups (mirror-pair buckets)
    mirror_map = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    vtrac_id = lambda d: min(d, mirror_map[d])  # 0..4
    vtrac_unique = len({vtrac_id(d) for d in digs})

    # mirror pairs any?
    mirror_pairs_any = "yes" if any(mirror_map[d] in digs for d in set(digs)) else "no"

    # unique count & repeat digit
    unique_count = len(set(digs))
    repeat_digit = "none"
    if unique_count < 4:
        # repeated digit value (string) for first repeated
        for d in digs:
            if digs.count(d) > 1:
                repeat_digit = str(d)
                break

    return {
        "seed_lead_band": lead_band,
        "seed_trail_band": trail_band,
        "seed_sum_lastdigit": str(s % 10),
        "seed_sum_band": sum_band,
        "seed_spread_band": spread_band,
        "seed_consecutive_pairs": str(consec),
        "seed_prime_count": str(prime_count),
        "seed_vtrac_unique": str(vtrac_unique),
        "seed_unique_count": str(unique_count),
        "seed_repeat_digit": repeat_digit,
        "seed_mirror_pairs_any": mirror_pairs_any,
    }


def _cold_due_member_flags_from_core_hits(df_all: pd.DataFrame, core: str, stream: str, asof_date: datetime.date, windows=(10, 15, 20)) -> Dict[int, Dict[str, bool]]:
    """For each window N, return {member_label: is_cold_due} using core-hit history in the same stream.

    cold = least-frequent member(s) in last N core-hit events (ties allowed)
    due  = member did NOT hit in last 2 core-hit events
    """
    flags: Dict[int, Dict[str, bool]] = {}
    try:
        core = str(core).zfill(3)
        members = members_from_core(core)  # list[str] 4-digit
        label_by_member = {m: structure_of_4(m) for m in members}
        member_by_label = {structure_of_4(m): m for m in members}

        # history of core-family hits in this stream before asof_date
        hist = df_all[(df_all["Stream"].astype(str) == str(stream)) & (df_all["Date"] < asof_date)].copy()
        if hist.empty:
            return {N: {lbl: False for lbl in ("AABC","ABBC","ABCC")} for N in windows}
        hist["Result4"] = hist["Result"].astype(str).apply(lambda x: extract_4digit(x) or str(x))
        core_hit = hist[hist["Result4"].isin(members)].sort_values("Date")
        if core_hit.empty:
            return {N: {lbl: False for lbl in ("AABC","ABBC","ABCC")} for N in windows}

        last2 = core_hit.tail(2)["Result4"].tolist()
        due_labels = {label_by_member[m] for m in members if m not in last2}

        for N in windows:
            tail = core_hit.tail(int(N))
            counts = {label_by_member[m]: int((tail["Result4"] == m).sum()) for m in members}
            min_ct = min(counts.values()) if counts else 0
            cold_labels = {lbl for lbl, ct in counts.items() if ct == min_ct}
            flags[N] = {lbl: (lbl in cold_labels) and (lbl in due_labels) for lbl in ("AABC","ABBC","ABCC")}
    except Exception:
        flags = {N: {lbl: False for lbl in ("AABC","ABBC","ABCC")} for N in windows}
    return flags


def _apply_member_traits_soft_override(
    base_probs: Dict[str, float],
    core: str,
    seed: str,
    stream: str,
    test_date: datetime.date,
    df_all: pd.DataFrame,
    member_pos_lut: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    member_neg_lut: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    weight: float,
) -> Tuple[str, List[str]]:
    """Return (top1_label, top2_labels) after soft member-trait adjustment."""
    core = str(core).zfill(3)
    # Ensure we have the as-of seed for this stream/date (needed for member-trait rules)
    if not seed:
        try:
            seed = _seed_for_stream_asof(df_all, stream, pd.to_datetime(test_date).normalize())
        except Exception:
            pass
    seed_traits = {}
    try:
        seed_traits.update(_seed_traits_for_core_stream(df_all, core, stream, pd.to_datetime(test_date).normalize()))
    except Exception:
        pass
    seed_traits.update(_seed_traits_extended(seed))

    # special cold/due flags per label per window
    cold_due_flags = _cold_due_member_flags_from_core_hits(df_all, core, stream, test_date, windows=(10, 15, 20))

    # adjust
    adj: Dict[str, float] = {}
    for lbl, p in base_probs.items():
        if p <= 0:
            continue
        mult = 1.0

        # positive rules
        for trait, vdict in member_pos_lut.get(core, {}).get(lbl, {}).items():
            for val, lift in vdict.items():
                if trait == "cold_due_member_group":
                    # fires if this label is cold+due under the window
                    try:
                        N = int(str(val).replace("N=", "").strip())
                        if cold_due_flags.get(N, {}).get(lbl, False):
                            mult *= float(lift)
                    except Exception:
                        continue
                else:
                    if seed_traits.get(trait, None) == val:
                        mult *= float(lift)

        # negative rules
        for trait, vdict in member_neg_lut.get(core, {}).get(lbl, {}).items():
            for val, lift in vdict.items():
                if trait == "cold_due_member_group":
                    try:
                        N = int(str(val).replace("N=", "").strip())
                        if cold_due_flags.get(N, {}).get(lbl, False):
                            mult *= float(lift)
                    except Exception:
                        continue
                else:
                    if seed_traits.get(trait, None) == val:
                        mult *= float(lift)

        # soft blend (exponentiate by weight)
        try:
            mult = float(mult) ** float(weight)
        except Exception:
            mult = 1.0
        adj[lbl] = p * mult

    if not adj:
        # fall back to base
        ordered = sorted(base_probs.items(), key=lambda kv: (-kv[1], kv[0]))
        top1 = ordered[0][0] if ordered else ""
        top2 = [kv[0] for kv in ordered[:2]]
        return top1, top2

    ordered = sorted(adj.items(), key=lambda kv: (-kv[1], kv[0]))
    top1 = ordered[0][0]
    top2 = [kv[0] for kv in ordered[:2]]
    return top1, top2


def _seed_sum_range4_labels(seed_sum: int) -> List[str]:
    labels = []
    for start in (seed_sum, seed_sum-1, seed_sum-2, seed_sum-3):
        if start < 0:
            continue
        end = start + 3
        if end > 36:  # pick4 max sum
            continue
        labels.append(f"{start}-{end}")
    return labels

def _seed_spread_bucket(spread: int) -> str:
    if spread <= 2:
        return "<=2"
    if spread >= 6:
        return ">=6"
    return "3-5"

def _count_high_digits(digits: List[int]) -> int:
    # Pick-4 convention in this app: high digits are 5–9
    return sum(1 for d in digits if d >= 5)

def _seed_contains_core_pair(seed: str, core: str) -> str:
    seed = str(seed)
    core = canonical_core_key(core)
    core_digits = list(core)
    pairs = set()
    for i in range(len(core_digits)):
        for j in range(len(core_digits)):
            if i == j:
                continue
            pairs.add(core_digits[i] + core_digits[j])
    adj = [seed[i:i+2] for i in range(len(seed)-1)]
    return "yes" if any(a in pairs for a in adj) else "no"

def _feature_values_for_seed(seed: str, core: str, last5_union_digits: Optional[set] = None) -> Dict[str, List[str]]:
    seed = str(seed).zfill(4)
    core = canonical_core_key(core)
    digits = [int(ch) for ch in seed]
    core_set = set(core)

    seed_sum = sum(digits)
    even_ct = sum(1 for d in digits if d % 2 == 0)
    high_ct = _count_high_digits(digits)
    spread = max(digits) - min(digits) if digits else 0

    overlap = len(set(seed) & core_set)  # unique overlap
    overlap_vals: List[str]
    if overlap == 3:
        overlap_vals = ["3", ">=2"]
    elif overlap == 2:
        overlap_vals = [">=2"]
    elif overlap == 0:
        overlap_vals = ["0"]
    else:
        overlap_vals = [str(overlap)]

    grid_overlap = 0
    if last5_union_digits:
        try:
            grid_overlap = len(set(last5_union_digits) & core_set)
        except Exception:
            grid_overlap = 0
    if grid_overlap == 3:
        grid_vals = ["3", ">=2"]
    elif grid_overlap == 2:
        grid_vals = [">=2"]
    else:
        grid_vals = [str(grid_overlap)]

    feats: Dict[str, List[str]] = {
        "seed_structure": [structure_of_4(seed)],
        "seed_even_count": [str(even_ct)],
        "seed_high_count": [str(high_ct)],
        "seed_spread": [_seed_spread_bucket(spread)],
        "seed_sum_mod2": [str(seed_sum % 2)],
        "seed_sum_mod3": [str(seed_sum % 3)],
        "seed_sum_range4_best": _seed_sum_range4_labels(seed_sum),
        "seed_sum_range4_worst": _seed_sum_range4_labels(seed_sum),
        "overlap_unique": overlap_vals,
        "seed_contains_core_pair": [_seed_contains_core_pair(seed, core)],
        "seed_first_in_core": ["yes" if seed[0] in core_set else "no"],
        "seed_last_in_core": ["yes" if seed[-1] in core_set else "no"],
        "grid_last5_core_digits": grid_vals,
    }
    return feats

def compute_seed_traits_score(
    core: str,
    seed: Optional[str],
    stream: Optional[str],
    *,
    pos_lookup: Dict[str, Dict[str, Dict[str, float]]],
    neg_lookup: Dict[str, Dict[str, Dict[str, float]]],
    last5_union_digits_by_stream: Optional[Dict[str, set]] = None,
    cap: float = 2.0,
    weight_pos: float = 1.0,
    weight_neg: float = 1.0,
) -> Tuple[float, List[Tuple[str, str, float, str]]]:
    """Return (net_score, matches). net_score is sum((lift-1) pos) - sum((lift-1) neg), capped."""
    if seed is None:
        return 0.0, []
    core = canonical_core_key(core)
    core_key = core
    last5_union = None
    if stream and last5_union_digits_by_stream and stream in last5_union_digits_by_stream:
        last5_union = last5_union_digits_by_stream.get(stream)
    feats = _feature_values_for_seed(str(seed), core, last5_union_digits=last5_union)

    matches: List[Tuple[str, str, float, str]] = []
    score = 0.0
    for trait, vals in feats.items():
        for val in vals:
            # positive
            liftp = pos_lookup.get(core_key, {}).get(trait, {}).get(val)
            if liftp is not None:
                delta = float(liftp) - 1.0
                score += weight_pos * delta
                matches.append((trait, val, float(liftp), "+"))
            # negative
            liftn = neg_lookup.get(core_key, {}).get(trait, {}).get(val)
            if liftn is not None:
                delta = float(liftn) - 1.0
                score -= weight_neg * delta
                matches.append((trait, val, float(liftn), "-"))
    # Cap for safety
    score = max(-cap, min(cap, score))
    return float(score), matches

def compute_cadence_score(days_since_last_hit: float, mean_gap_days: float) -> float:
    """Soft cadence score in [0,1]. 0 = not due vs cadence, 1 = very due."""
    try:
        d = float(days_since_last_hit)
    except Exception:
        return 0.0
    try:
        g = float(mean_gap_days)
    except Exception:
        g = 0.0
    if g <= 0:
        return 0.0
    ratio = d / g
    # Map ratio: 1.0 -> 0, 3.0 -> 1 (cap)
    val = (ratio - 1.0) / 2.0
    if val < 0:
        return 0.0
    if val > 1:
        return 1.0
    return float(val)


def get_position_percentiles_cached(core: str, window_days: int, stream_stats: pd.DataFrame) -> pd.DataFrame:
    """Cache position percentile maps per core/window so UI tweaks don't constantly recompute.
    Cache is automatically cleared when input data changes or when the user clicks 'Recompute percentile maps now'.
    """
    cache: Dict[str, pd.DataFrame] = st.session_state.get("pos_map_cache", {})
    data_hash = st.session_state.get("data_hash_all", "")
    key = f"{core}|{window_days}|{data_hash}"

    if key in cache:
        return cache[key]

    pos_map, _ = position_percentile_map(stream_stats)
    cache[key] = pos_map
    st.session_state["pos_map_cache"] = cache

    if not st.session_state.get("recompute_token"):
        st.session_state["recompute_token"] = datetime.datetime.now().isoformat(timespec="seconds")

    return pos_map

def bucket_recommendations(
    stream_stats: pd.DataFrame,
    cfg: Optional[RankConfig] = None,
    *,
    top_n: Optional[int] = None,
    due_n: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Build Northern Star buckets from a stream_stats table.

    Returns a dict with **multiple key aliases** for compatibility:
      - Top12BaseScore / Top12  -> top base-score bucket
      - Due8                   -> due bucket
      - Combined               -> merged bucket
      - base_top / due_top / combined -> lists of stream labels (for meta)

    `top_n` / `due_n` are accepted as legacy keyword overrides.
    """
    if cfg is None:
        cfg = RankConfig()

    base_n = int(top_n) if top_n is not None else int(getattr(cfg, "top_base", 12))
    due_take = int(due_n) if due_n is not None else int(getattr(cfg, "top_due", 8))

    # Normalize expected columns so this helper works with:
    #  - stream_summary() output (RankPos, HitsPerWeek, DaysSinceLastHit, ...)
    #  - legacy bucket tables (BaseRank/DueRank)
    #  - future member-level tables (Pick/Member columns)
    df = stream_stats.copy()

    # Base rank
    if "BaseScoreRank" not in df.columns:
        if "BaseRank" in df.columns:
            df["BaseScoreRank"] = df["BaseRank"]
        elif "RankPos" in df.columns:
            df["BaseScoreRank"] = df["RankPos"]
        elif "HitsPerWeek" in df.columns:
            _hpw = pd.to_numeric(df["HitsPerWeek"], errors="coerce").fillna(0.0)
            df["BaseScoreRank"] = (-_hpw).rank(method="dense", ascending=True).astype(int)
        else:
            df["BaseScoreRank"] = range(1, len(df) + 1)

    # RankPos (for display ordering)
    if "RankPos" not in df.columns:
        df["RankPos"] = df["BaseScoreRank"]

    # Due index
    if "DueIndex" not in df.columns:
        if "DaysSinceLastHit" in df.columns:
            df["DueIndex"] = pd.to_numeric(df["DaysSinceLastHit"], errors="coerce")
        else:
            df["DueIndex"] = 0

    # Due rank
    if "DueIndexRank" not in df.columns:
        if "DueRank" in df.columns:
            df["DueIndexRank"] = df["DueRank"]
        else:
            _di = pd.to_numeric(df["DueIndex"], errors="coerce").fillna(-1)
            # 1 = most due (largest DueIndex)
            df["DueIndexRank"] = (-_di).rank(method="dense", ascending=True).astype(int)

    # Ensure Stream exists for downstream logic
    if "Stream" not in df.columns and "stream" in df.columns:
        df["Stream"] = df["stream"]
    due_lo = int(getattr(cfg, "due_from_rank", 13))
    due_hi = int(getattr(cfg, "due_to_rank", 60))

    # Defensive: tolerate missing columns; caller should validate upstream.
    if stream_stats is None or len(df) == 0:
        empty = pd.DataFrame()
        return {
            "Top12BaseScore": empty,
            "Top12": empty,
            "Due8": empty,
            "Combined": empty,
            "base_top": [],
            "due_top": [],
            "combined": [],
        }

    base_df = df.sort_values("BaseScoreRank", ascending=True).head(base_n)
    due_pool = df[
        (df["BaseScoreRank"] >= due_lo) & (df["BaseScoreRank"] <= due_hi)
    ].sort_values("DueIndexRank", ascending=True)
    due_df = due_pool.head(due_take)

    combined_df = pd.concat([base_df, due_df], ignore_index=True).drop_duplicates(subset=["Stream"], keep="first")
    combined_df = combined_df.sort_values("RankPos", ascending=True)

    base_streams = base_df["Stream"].tolist() if "Stream" in base_df.columns else []
    due_streams = due_df["Stream"].tolist() if "Stream" in due_df.columns else []
    combined_streams = combined_df["Stream"].tolist() if "Stream" in combined_df.columns else []

    return {
        "Top12BaseScore": base_df,
        "Top12": base_df,
        "Due8": due_df,
        "Combined": combined_df,
        "base_top": base_streams,
        "due_top": due_streams,
        "combined": combined_streams,
    }

def build_northern_star_bucket_meta(
    stream_stats: pd.DataFrame,
    cfg: RankConfig,
    *,
    seed_core_key: str = "",
    include_24h: bool = False,
    df_24: pd.DataFrame | None = None,
    core: str = "",
) -> List[Dict[str, Any]]:
    """Compatibility helper used by some older app revisions.

    Returns a list of per-stream bucket metadata rows (one dict per stream).
    This intentionally mirrors the data shape consumed by the Northern Lights master playlist.
    """
    if stream_stats is None or not isinstance(stream_stats, pd.DataFrame) or stream_stats.empty:
        return []

    # Pre-compute which streams are in which bucket for this core.
    rec = bucket_recommendations(stream_stats, cfg)
    base_streams = set(rec.get("base_top", []))
    due_streams = set(rec.get("due_top", []))

    rows: List[Dict[str, Any]] = []
    for stream in stream_stats["Stream"].tolist():
        try:
            rows.append(
                build_northern_star_buckets(
                    stats_df=stream_stats,
                    stream=stream,
                    top_n=cfg.top_base,
                    due_ranks=(cfg.due_from_rank, cfg.due_to_rank),
                    seed_core_key=seed_core_key,
                    include_24h=include_24h,
                    df_24=df_24,
                    core=core,
                    base_streams=base_streams,
                    due_streams=due_streams,
                )
            )
        except TypeError:
            # Oldest signature (no precomputed sets)
            rows.append(
                build_northern_star_buckets(
                    stats_df=stream_stats,
                    stream=stream,
                    top_n=cfg.top_base,
                    due_ranks=(cfg.due_from_rank, cfg.due_to_rank),
                    seed_core_key=seed_core_key,
                    include_24h=include_24h,
                    df_24=df_24,
                    core=core,
                )
            )
    return rows

def top_dense_positions(pos_map, top_n: int = 10, top_k_positions: int | None = None):
    """Return the top-N rank positions (1..76) that collectively hold the most winners."""
    if pos_map is None or pos_map.empty:
        return []
    tmp = pos_map.copy()
    # Defensive: sometimes RankPos can come back as strings; coerce to numeric
    tmp["RankPos_num"] = pd.to_numeric(tmp["RankPos"], errors="coerce")
    tmp = tmp.dropna(subset=["RankPos_num"])
    if tmp.empty:
        return []
    tmp["RankPos_num"] = tmp["RankPos_num"].astype(int)
    counts = tmp.groupby("RankPos_num")["HitCount"].sum().sort_values(ascending=False).head(int(top_n))
    return [int(x) for x in counts.index.tolist()]

def engine_cluster_positions(
    df_24h_core_hits: pd.DataFrame,
    base_stats: pd.DataFrame,
    top_n: int = 10,
    use_rank_col: str = "RankPos",
) -> list[int]:
    """Return the *clustered* top-N rank positions for a 24h core-hit sample.

    Robust against empty inputs, NaNs, and callers accidentally passing Series/dicts.
    Always returns a Python list (possibly empty).
    """
    if df_24h_core_hits is None or len(df_24h_core_hits) == 0:
        return []

    if base_stats is None:
        return []

    # Normalize base_stats to a DataFrame with a numeric rank column.
    if not isinstance(base_stats, pd.DataFrame):
        try:
            base_stats = pd.DataFrame(base_stats)
        except Exception:
            return []

    if use_rank_col not in base_stats.columns:
        return []

    rank_input = base_stats[use_rank_col]
    if isinstance(rank_input, pd.DataFrame):
        # if a list-like column selector was passed, take first column
        rank_input = rank_input.iloc[:, 0]
    rank_series = pd.to_numeric(rank_input, errors="coerce")
    rank_map = pd.DataFrame({"RankPos": rank_series}).dropna().sort_values("RankPos")
    if rank_map.empty:
        return []

    # Candidate rank positions actually observed in the 24h hit set
    try:
        hit_ranks = pd.to_numeric(df_24h_core_hits.get("RankPos"), errors="coerce").dropna().astype(int).tolist()
    except Exception:
        hit_ranks = []

    if not hit_ranks:
        return []

    # Keep only ranks that exist in base_stats map
    rank_set = set(rank_map["RankPos"].astype(int).tolist())
    hit_ranks = [int(r) for r in hit_ranks if int(r) in rank_set]
    if not hit_ranks:
        return []

    # Find a "dense cluster" around the most common local neighborhood.
    hit_ranks_sorted = sorted(hit_ranks)

    best_window = None
    best_score = -1
    span = 12  # neighborhood width

    for anchor in hit_ranks_sorted:
        lo = anchor
        hi = anchor + span
        members = [r for r in hit_ranks_sorted if lo <= r <= hi]
        score = len(members)
        if score > best_score:
            best_score = score
            best_window = (lo, hi)

    if best_window is None:
        return []

    lo, hi = best_window
    clustered = [r for r in rank_map["RankPos"].astype(int).tolist() if lo <= r <= hi]
    # Return up to top_n, but keep as list[int]
    return clustered[: max(1, int(top_n))]


def evaluate_rare_engine(
    df_all: pd.DataFrame,
    core: str,
    df_24h: pd.DataFrame | None = None,
    enable_r1: bool = True,
    enable_r2: bool = True,
    enable_r3: bool = True,
    enable_r4: bool = True,
    window_days_recent: int = 180,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if df_24h is None:
        df_24h = pd.DataFrame()
    """
    Rare Engine checks (AAAB + AABB together):
      R1: stream is in top 20% for combined AAAB+AABB baseline rate (full history).
      R2: stream is in top 20% for combined AAAB+AABB rate in last 180 days.
      R3: the last 24h map contains ≥3 AAAB/AABB hits across ≥3 distinct streams (global condition).
      R4: last 24h AAAB/AABB hits cluster into Top-10 RankPos, and stream RankPos is in that set.
    Trigger: at least 3 of enabled checks True.

    Returns:
      - per-stream table with booleans and trigger
      - summary dict with thresholds and cluster sets
    """
    if df_all.empty:
        return pd.DataFrame(), {"error":"No history loaded."}

    core = canonical_core_key(core)
    df_hits_all = compute_core_hits(df_all, core, structures=["AAAB","AABB"])

    # Baseline stream stats
    base_stats = stream_summary(df_all, df_hits_all, window_days=min(365*5, int((df_all["Date"].max()-df_all["Date"].min()).days) or 365))
    # But we want baseline based on full history span; use hits per week in that span:
    span_days = max(int((df_all["Date"].max()-df_all["Date"].min()).days), 1)
    base_stats["HitsPerWeek_full"] = base_stats["HitsWindow"] / (span_days/7.0)
    base_stats = base_stats.sort_values(["HitsPerWeek_full","HitsWindow"], ascending=False).reset_index(drop=True)
    base_stats["RankPos_full"] = np.arange(1, len(base_stats)+1)

    # Recent stats (180d)
    recent_stats = stream_summary(df_all, df_hits_all, window_days=window_days_recent)

    # Thresholds
    def pct_threshold(series: pd.Series, pct: float) -> float:
        vals = series.dropna().values
        if len(vals)==0:
            return float("nan")
        return float(np.quantile(vals, pct))

    # R1 top 20% based on HitsPerWeek_full
    thr_r1 = pct_threshold(base_stats["HitsPerWeek_full"], 0.80)

    # R2 top 20% based on recent HitsPerWeek
    thr_r2 = pct_threshold(recent_stats["HitsPerWeek"], 0.80)

    # R3 global condition from 24h file: ≥3 hits across ≥3 distinct streams
    df_24h_core_hits = pd.DataFrame()
    top10_cluster = []
    r3_global = False
    if df_24h is not None and not df_24h.empty:
        df_24h_core_hits = compute_core_hits(df_24h, core, structures=["AAAB","AABB"])
        n_hits_24h = int(len(df_24h_core_hits))
        n_streams_24h = int(df_24h_core_hits["Stream"].nunique()) if n_hits_24h else 0
        r3_global = (n_hits_24h >= 3) and (n_streams_24h >= 3)
        # R4 cluster based on 24h file (Top-10 RankPos positions by 24h frequency)
        top10_cluster = engine_cluster_positions(
            df_24h_core_hits,
            base_stats.rename(columns={"RankPos_full":"RankPos"}).assign(RankPos=lambda d: d["RankPos"]),
            top_n=10,
        )

    # Merge per stream
    out = pd.DataFrame({"Stream": base_stats["Stream"]})
    out = out.merge(base_stats[["Stream","HitsPerWeek_full","RankPos_full"]], on="Stream", how="left")
    out = out.merge(recent_stats[["Stream","HitsPerWeek","DaysSinceLastHit","RankPos"]].rename(columns={"RankPos":"RankPos_recent"}), on="Stream", how="left")

    out["R1_top20_baseline"] = out["HitsPerWeek_full"] >= thr_r1 if enable_r1 else False
    out["R2_top20_recent"] = out["HitsPerWeek"] >= thr_r2 if enable_r2 else False
    out["R3_24h_has_3plus_across_3streams"] = r3_global if enable_r3 else False
    out["R4_24h_cluster_top10pos"] = out["RankPos_full"].isin(top10_cluster) if enable_r4 else False

    enabled_cols = [c for c, en in [
        ("R1_top20_baseline", enable_r1),
        ("R2_top20_recent", enable_r2),
        ("R3_24h_has_3plus_across_3streams", enable_r3),
        ("R4_24h_cluster_top10pos", enable_r4),
    ] if en]

    out["ChecksTrue"] = out[enabled_cols].sum(axis=1) if enabled_cols else 0
    out["RareEngine_TRIG"] = out["ChecksTrue"] >= 3 if enabled_cols else False

    out = out.sort_values(["RareEngine_TRIG","ChecksTrue","HitsPerWeek_full"], ascending=[False, False, False]).reset_index(drop=True)

    summary = {
        "thr_r1": thr_r1,
        "thr_r2": thr_r2,
        "r3_global": r3_global,
        "top10_cluster_positions": top10_cluster,
        "n_24h_core_hits": int(len(df_24h_core_hits)) if df_24h is not None else 0,
        "n_24h_core_streams": int(df_24h_core_hits["Stream"].nunique()) if df_24h is not None and not df_24h_core_hits.empty else 0,
        "span_days_full": span_days,
        "enabled_checks": enabled_cols,
    }
    return out, summary

def evaluate_ultra_rare_engine(
    df_all: pd.DataFrame,
    core: str,
    df_24h: pd.DataFrame | None = None,
    enable_q1: bool = True,
    enable_q2: bool = True,
    enable_q3: bool = True,
    enable_q4: bool = True,
    # Some UI call-sites pass this (mirroring the rare engine). We accept it for
    # compatibility. The ultra-rare engine is primarily computed on the full
    # history; when provided, we use it only for optional recent-window fields.
    window_days_recent: int | None = None,
    # Forward-compat: ignore any extra kwargs passed from older/newer UIs.
    **_ignored_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Ultra-Rare Engine checks (AAAA quads for the core's digits):
      Q1: stream in top 10% for quad baseline rate (full history)
      Q2: days since last quad >= 90th percentile across streams
      Q3: last 24h has at least 1 quad anywhere (for any digit in core)
      Q4: 24h quad hits (for core) cluster into Top-5 RankPos; stream position is in that set
    Trigger: at least 2 of enabled checks True.
    """
    if df_24h is None:
        df_24h = pd.DataFrame()

    if df_all.empty:
        return pd.DataFrame(), {"error":"No history loaded."}

    core = canonical_core_key(core)
    df_hits_all = compute_core_hits(df_all, core, structures=["AAAA"])

    span_days = max(int((df_all["Date"].max()-df_all["Date"].min()).days), 1)
    base_stats = stream_summary(df_all, df_hits_all, window_days=min(365*5, span_days))

    # Optional recent window (used by some UI displays). This is non-breaking: if
    # absent, downstream behavior matches historical-only mode.
    base_stats_recent = None
    if window_days_recent is not None and not df_all.empty and "Date" in df_all.columns:
        try:
            cutoff = df_all["Date"].max() - pd.Timedelta(days=int(window_days_recent))
            df_recent = df_all[df_all["Date"] >= cutoff].copy()
            df_hits_recent = compute_core_hits(df_recent, core, structures=["AAAA"])
            base_stats_recent = stream_summary(df_recent, df_hits_recent, window_days=int(window_days_recent))
        except Exception:
            base_stats_recent = None
    base_stats["HitsPerWeek_full"] = base_stats["HitsWindow"] / (span_days/7.0)
    base_stats = base_stats.sort_values(["HitsPerWeek_full","HitsWindow"], ascending=False).reset_index(drop=True)
    base_stats["RankPos_full"] = np.arange(1, len(base_stats)+1)

    # thresholds
    def pct_threshold(series: pd.Series, pct: float) -> float:
        vals = series.dropna().values
        if len(vals)==0:
            return float("nan")
        return float(np.quantile(vals, pct))

    thr_q1 = pct_threshold(base_stats["HitsPerWeek_full"], 0.90)
    thr_q2 = pct_threshold(base_stats["DaysSinceLastHit"], 0.90)  # DaysSinceLastHit computed from last quad date

    # Q3 global 24h quad exists for any core digit
    q3_global = False
    df_24h_core_hits = pd.DataFrame()
    top5_cluster = []
    if df_24h is not None and not df_24h.empty:
        # any quad in 24h that uses one of core digits
        core_digits = set(core)
        df_24h_quads = df_24h[df_24h["Structure"]=="AAAA"].copy()
        df_24h_quads["quad_digit"] = df_24h_quads["Pick4"].str[0]
        q3_global = df_24h_quads["quad_digit"].isin(core_digits).any()
        df_24h_core_hits = compute_core_hits(df_24h, core, structures=["AAAA"])
        top5_cluster = engine_cluster_positions(df_24h_core_hits, base_stats.rename(columns={"RankPos_full":"RankPos"}).assign(RankPos=lambda d: d["RankPos"]), top_n=5)

    out = pd.DataFrame({"Stream": base_stats["Stream"]})
    out = out.merge(base_stats[["Stream","HitsPerWeek_full","DaysSinceLastHit","RankPos_full"]], on="Stream", how="left")

    out["Q1_top10_baseline"] = out["HitsPerWeek_full"] >= thr_q1 if enable_q1 else False
    out["Q2_due_p90"] = out["DaysSinceLastHit"] >= thr_q2 if enable_q2 else False
    out["Q3_24h_quad_exists"] = q3_global if enable_q3 else False
    out["Q4_24h_cluster_top5pos"] = out["RankPos_full"].isin(top5_cluster) if enable_q4 else False

    enabled_cols = [c for c, en in [
        ("Q1_top10_baseline", enable_q1),
        ("Q2_due_p90", enable_q2),
        ("Q3_24h_quad_exists", enable_q3),
        ("Q4_24h_cluster_top5pos", enable_q4),
    ] if en]

    out["ChecksTrue"] = out[enabled_cols].sum(axis=1) if enabled_cols else 0
    out["UltraRare_TRIG"] = out["ChecksTrue"] >= 2 if enabled_cols else False

    out = out.sort_values(["UltraRare_TRIG","ChecksTrue","HitsPerWeek_full"], ascending=[False, False, False]).reset_index(drop=True)

    summary = {
        "thr_q1": thr_q1,
        "thr_q2": thr_q2,
        "q3_global": q3_global,
        "top5_cluster_positions": top5_cluster,
        "n_24h_core_quad_hits": int(len(df_24h_core_hits)) if df_24h is not None else 0,
        "enabled_checks": enabled_cols,
        "span_days_full": span_days,
    }
    return out, summary


# -------------------------
# UI
# -------------------------


def render_backtest(
    df_all: pd.DataFrame,
    cfg: "RankConfig | None" = None,
    cores_for_cache: "list[str] | None" = None,
    df_24h: "pd.DataFrame | None" = None,
    # backwards-compatible aliases (older call sites)
    cores: "list[str] | None" = None,
    window_days: "int | None" = None,
):
    """Backtest / diagnostics.

    **Walk-forward mode (no-cheat):**
    For each test_date, builds rankings/traps using ONLY rows with Date < test_date, then scores the
    winner(s) that occurred on test_date.

    **Playlist diagnostic mode:**
    Uses the current Northern Lights playlist in-session (helpful for quick validation, but can
    include future leakage if the playlist was built using the full dataset).
    """

    # Normalize args (avoid brittle keyword mismatches across revisions)
    if cfg is None:
        cfg = RankConfig(window_days=int(window_days or 180))
    if cores_for_cache is None:
        cores_for_cache = list(cores or [])

    st.subheader("Backtest (optional)")
    st.caption("Optional diagnostics. Walk-forward mode avoids future leakage by training only on Date < test_date.")

    if df_all is None or getattr(df_all, "empty", True):
        st.warning("Upload an all-states history file to use Backtest.")
        return

    # Ensure Date dtype
    if "Date" in df_all.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_all["Date"]):
                df_all = df_all.copy()
                df_all["Date"] = _parse_date_series(df_all["Date"])
        except Exception:
            pass

    mode = st.radio(
        "Backtest mode",
        ["Walk-forward (no cheating)", "Playlist diagnostic (uses current playlist)"],
        horizontal=True,
        key="bt_mode_v51",
    )

    # Clarify which "view" drives stream selection to prevent confusion.
    if mode.startswith("Walk-forward"):
        st.info("Stream-bucket source: WALK-FORWARD per-core recompute (train on Date < test_date; uses your selected 180/365 window).")
    else:
        st.info("Stream-bucket source: CURRENT Northern Lights playlist on-screen (playlist diagnostic; not walk-forward).")


    if mode.startswith("Walk-forward"):
        _render_backtest_walk_forward(df_all=df_all, cfg=cfg, cores_for_cache=cores_for_cache)
        return

    # Playlist diagnostic (legacy / quick)
    """Optional diagnostics backtest.

    This evaluates how often the *current* Northern Lights master playlist (streams+cores)
    would have caught a matching family member in those streams over a selected historical range.
    It does **not** change any scoring or ranking output.
    """

    if df_all is None or df_all.empty:
        st.warning("Upload an all-states history file first.")
        return

    # Ensure required columns exist
    if "Date" not in df_all.columns or "Stream" not in df_all.columns:
        st.error("History file is missing required columns (Date, Stream).")
        return

    # Date bounds
    try:
        dmin_ts = pd.to_datetime(df_all["Date"]).min()
        dmax_ts = pd.to_datetime(df_all["Date"]).max()
    except Exception:
        st.error("Could not read Date values from the history file.")
        return

    if pd.isna(dmin_ts) or pd.isna(dmax_ts):
        st.error("History file has no valid dates.")
        return

    dmin = dmin_ts.date()
    dmax = dmax_ts.date()
    default_start = max(dmin, (dmax_ts - pd.Timedelta(days=min(180, max(7, int((dmax_ts - dmin_ts).days * 0.25))))).date())
    date_range = st.date_input(
        "Backtest date range (inclusive)",
        value=(default_start, dmax),
        min_value=dmin,
        max_value=dmax,
        help="This checks historical draws in the selected range against your current playlist picks."
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range, date_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # Structures used to define a 'hit' for a core in a stream
    structure_mode = st.selectbox(
        "Match mode (what counts as a hit for a core)",
        options=[
            "AABC only (single-member focus)",
            "Family mode (AABC + ABBC + ABCC)",
            "Rare engine mode (AAAB + AABB)",
            "Ultra-rare mode (AAAA only)",
        ],
        index=1,
        help="This affects only the backtest match check (not your ranking)."
    )
    if structure_mode.startswith("AABC only"):
        structures = ["AABC"]
    elif structure_mode.startswith("Family mode"):
        structures = ["AABC", "ABBC", "ABCC"]
    elif structure_mode.startswith("Rare engine mode"):
        structures = ["AAAB", "AABB"]
    else:
        structures = ["AAAA"]

    # Playlist source
    nl_df = st.session_state.get("nl_df_current")
    if not isinstance(nl_df, pd.DataFrame) or nl_df.empty or "Core" not in nl_df.columns or "Stream" not in nl_df.columns:
        with st.expander("Master playlist not found in session — build it now", expanded=True):
            st.info("Your master playlist isn't cached in the current session. Click below to build it (this can take a bit).")
            if st.button("Build master playlist for backtest", type="primary"):
                nl_df = None  # force rebuild below

    def _build_master_playlist_for_backtest() -> pd.DataFrame:
        cores = [canonical_core_key(c) for c in (cores_for_cache or [])]
        cores = [c for c in cores if c]
        if not cores:
            # Fall back to whatever is selected in session state
            cores = [canonical_core_key(c) for c in st.session_state.get("cores_selected", [])]
            cores = [c for c in cores if c]
        if not cores:
            return pd.DataFrame()

        cache = _load_baseline_from_disk(cfg.window_days)
        cache_ok = False
        if isinstance(cache, dict):
            cached_cores = set(cache.get("cores", []) or [])
            if cached_cores and all(c in cached_cores for c in cores):
                cache_ok = True

        rows = []
        progress = st.progress(0)
        for i, core in enumerate(cores, start=1):
            if cache_ok:
                stream_stats = cache["core_stream_stats"][core]
                pos_map = cache["core_pos_maps"][core]
            else:
                stream_stats = compute_stream_stats(df_all, core_key=core, structures=("AABC",), window_days=cfg.window_days)
                last_date = None
                try:
                    if 'Date' in df_all.columns:
                        last_date = str(pd.to_datetime(df_all['Date']).max().date())
                except Exception:
                    last_date = None
                pos_map = get_pos_map_cached(core, cfg.window_days, stream_stats, last_date)
            bucket_rows = build_northern_star_buckets(stream_stats, pos_map, cfg)
            bucket_rows = bucket_rows.copy()
            bucket_rows["Core"] = core
            rows.append(bucket_rows)
            progress.progress(int(i / max(1, len(cores)) * 100))
        progress.empty()

        if not rows:
            return pd.DataFrame()

        out = pd.concat(rows, ignore_index=True)

        # Universal score (keep identical to Northern Lights tab)
        out = out.copy()
        out["RecentStrength"] = out["RecentHitRate"] * 100.0
        out["DuePressure"] = out["DueScore"]
        out["PosStrength"] = out["PosPctScore"]
        out["UniversalScore"] = 0.45 * out["RecentStrength"] + 0.35 * out["DuePressure"] + 0.20 * out["PosStrength"]
        out["UniversalRank"] = out["UniversalScore"].rank(ascending=False, method="min").astype(int)

        # Column order preference
        cols_front = [
            "UniversalRank", "UniversalScore", "Core", "Stream",
            "Bucket", "BucketPick", "BaseRank", "DueRank",
            "RecentHitRate", "DueScore", "PosPctScore",
            "DaysSinceLastHit", "HitsWindow", "DrawsWindow"
        ]
        out = out[[c for c in cols_front if c in out.columns] + [c for c in out.columns if c not in cols_front]]
        out = out.sort_values(["UniversalRank", "Core", "Stream"]).reset_index(drop=True)
        return out

    if nl_df is None:
        nl_df = _build_master_playlist_for_backtest()
        if nl_df is not None and not nl_df.empty:
            st.session_state["nl_df_current"] = nl_df

    if not isinstance(nl_df, pd.DataFrame) or nl_df.empty:
        st.warning("No master playlist available to backtest. Build it first (Northern Lights tab or button above).")
        return

    playlist_mode = st.radio(
        "What to backtest",
        options=["Top N overall (by UniversalScore)", "Top 1 per stream (best core per stream)", "All playlist entries"],
        index=0,
        horizontal=True,
    )

    # Build the picks table (Core + Stream)
    nl_df_sorted = nl_df.sort_values(["UniversalScore", "UniversalRank"], ascending=[False, True]).copy()
    if playlist_mode.startswith("Top N overall"):
        max_n = max(1, min(500, int(len(nl_df_sorted))))
        default_n = min(39, max_n)
        top_n = st.slider("Top N entries to play each day", 1, max_n, default_n)
        picks = nl_df_sorted.head(top_n)[["Core", "Stream"]].dropna().drop_duplicates().reset_index(drop=True)
    elif playlist_mode.startswith("Top 1 per stream"):
        picks = (
            nl_df_sorted.sort_values(["Stream", "UniversalScore"], ascending=[True, False])
            .groupby("Stream", as_index=False)
            .head(1)[["Core", "Stream"]]
            .dropna().drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        picks = nl_df_sorted[["Core", "Stream"]].dropna().drop_duplicates().reset_index(drop=True)

    if picks.empty:
        st.warning("No playlist picks available for the selected backtest mode.")
        return

    # Cost settings (optional)
    st.markdown("#### Cost assumptions (optional)")
    colc1, colc2, colc3 = st.columns([1, 1, 2])
    with colc1:
        cost_per_play = st.number_input("Cost per play", min_value=0.0, value=0.25, step=0.05)
    with colc2:
        payout_per_win = st.number_input("Payout per win", min_value=0.0, value=247.50, step=1.0)
    with colc3:
        if structure_mode.startswith("Family mode"):
            default_numbers = 3
        else:
            default_numbers = 1
        numbers_per_pick = st.number_input(
            "Number of box numbers per (Core+Stream) pick",
            min_value=1,
            value=int(default_numbers),
            step=1,
            help="If you play all members of a family core, set this to 3. If you play only one number per pick, set to 1."
        )

    # Filter history to date range
    df_range = df_all.copy()
    df_range["Date"] = pd.to_datetime(df_range["Date"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_range = df_range[(df_range["Date"] >= start_ts) & (df_range["Date"] <= end_ts)]

    if df_range.empty:
        st.warning("No draws found in the selected date range.")
        return

    # Ensure BoxKey4 exists
    if "BoxKey4" not in df_range.columns:
        if "Result" not in df_range.columns:
            st.error("History file is missing Result/BoxKey4 needed for matching.")
            return
        df_range = df_range.copy()
        df_range["BoxKey4"] = df_range["Result"].astype(str).str.zfill(4).map(box_key)

    # Precompute draw counts per stream for opportunity counting
    draws_by_stream = df_range["Stream"].value_counts().to_dict()

    # Evaluate hits per pick
    core_to_streams = picks.groupby("Core")["Stream"].apply(list).to_dict()

    records = []
    total_opportunities = 0
    total_hits = 0
    total_unique_hit_days = set()

    for core, streams in core_to_streams.items():
        if not streams:
            continue
        members = set(members_from_core(core, structures=structures))
        if not members:
            continue

        df_s = df_range[df_range["Stream"].isin(streams)]
        if df_s.empty:
            continue

        df_hits = df_s[df_s["BoxKey4"].isin(members)]
        # opportunities = sum draws for each stream used by this core
        core_opps = sum(int(draws_by_stream.get(s, 0)) for s in streams)
        core_hits = int(len(df_hits))

        total_opportunities += core_opps
        total_hits += core_hits
        total_unique_hit_days.update(df_hits["Date"].dt.date.unique().tolist())

        # Stream-level breakdown
        hits_by_stream = df_hits.groupby("Stream").size().to_dict()
        for s in streams:
            opp = int(draws_by_stream.get(s, 0))
            h = int(hits_by_stream.get(s, 0))
            records.append({

"Core": core,
"ActualMemberLabel": None,
"ActualFamilyMember": None,
"PredMemberTop1": None,
"PredMemberTop2": None,
"MemberHitTop1": None,
"MemberHitTop2": None,
"MemberTrainN": 0,
"TrainCnt_AABC": 0,
"TrainCnt_ABBC": 0,
"TrainCnt_ABCC": 0,
                "Stream": s,
                "Opportunities": opp,
                "Hits": h,
                "HitRate": (h / opp) if opp else 0.0
            })

    if not records:
        st.warning("No matching hits found for the selected settings.")
        return

    bt_df = pd.DataFrame(records).sort_values(["Hits", "HitRate"], ascending=[False, False]).reset_index(drop=True)

    days_in_range = int((end_ts.normalize() - start_ts.normalize()).days) + 1
    plays_per_day = int(len(picks)) * int(numbers_per_pick)
    total_plays = int(total_opportunities) * int(numbers_per_pick)
    est_cost = float(total_plays) * float(cost_per_play)
    est_payout = float(total_hits) * float(payout_per_win)
    est_profit = est_payout - est_cost

    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Hits", f"{total_hits}")
    colm2.metric("Unique hit days", f"{len(total_unique_hit_days)} / {days_in_range}")
    colm3.metric("Opportunities", f"{total_opportunities}")
    colm4.metric("Plays/day (assumed)", f"{plays_per_day}")

    st.markdown("#### Estimated cost / payout (using your assumptions)")
    colp1, colp2, colp3 = st.columns(3)
    colp1.metric("Estimated cost", f"${est_cost:,.2f}")
    colp2.metric("Estimated payout", f"${est_payout:,.2f}")
    colp3.metric("Estimated profit", f"${est_profit:,.2f}")

    st.markdown("#### Backtest breakdown (per Core + Stream)")
    st.dataframe(bt_df, use_container_width=True)

    # Summaries
    st.markdown("#### Summary by core")
    core_sum = bt_df.groupby("Core", as_index=False).agg(
        Opportunities=("Opportunities", "sum"),
        Hits=("Hits", "sum")
    )
    core_sum["HitRate"] = core_sum["Hits"] / core_sum["Opportunities"].replace(0, np.nan)
    core_sum = core_sum.sort_values(["Hits", "HitRate"], ascending=[False, False]).reset_index(drop=True)
    st.dataframe(core_sum, use_container_width=True)

    # Download
    csv_bytes = bt_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download backtest breakdown (CSV)",
        data=csv_bytes,
        file_name="backtest_core_stream_breakdown.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -------------------------
# SAFETY DEFAULTS (prevent NameError on first load / before actions run)
# These are overwritten later when the relevant UI/actions execute.
# -------------------------
try:
    import pandas as _pd  # already imported as pd later; safe fallback
except Exception:
    _pd = None

out = (_pd.DataFrame() if _pd is not None else None)  # walk-forward output placeholder
member_track = False  # member tracking checkbox state placeholder
cores_for_cache = []  # multi-core list placeholder
cores_for_cache_ms = []  # multiselect placeholder
df_all = (_pd.DataFrame() if _pd is not None else None)
df_24h = (_pd.DataFrame() if _pd is not None else None)

st.set_page_config(page_title="Pick 4 Northern Star", layout="wide", initial_sidebar_state="expanded")



# Show build/version in UI (helps verify you deployed the right file)
try:
    st.sidebar.caption(f'Build: {APP_BUILD_ID} | base={APP_BUILD_BASE} | hash={APP_BUILD_HASH}')
except Exception:
    pass
st.title("Pick 4 — Northern Star + Rare Engine (AAAB+AABB) + Ultra‑Rare (AAAA)")

# Safe init for sidebar footer (values are filled after parsing uploads)
last_all = None
last_24 = None
df_all = None
df_24 = None


with st.sidebar:
    st.header("Data")
    master_file = st.file_uploader("All‑states history file (.csv or LotteryPost .txt)", type=["csv","txt"])
    map24_file = st.file_uploader("24h map file (optional, same format)", type=["csv","txt"])

    with st.expander("Troubleshooting (hard reset)", expanded=False):
        st.caption("If tables still show after you clear cache / refresh, use this to delete disk caches (baseline + pctmaps) and clear Streamlit caches.")

        def _on_hard_reset_click():
            actions = _hard_reset_disk_and_streamlit_caches()
            st.session_state["_hard_reset_actions"] = actions
            st.session_state["_hard_reset_nonce"] = time.time()

        st.button("Hard reset caches (disk + memory)", key="hard_reset_btn", on_click=_on_hard_reset_click)

        if st.session_state.get("_hard_reset_actions"):
            st.success("Hard reset complete:\n- " + "\n- ".join(st.session_state.get("_hard_reset_actions", [])))


    st.divider()
    st.subheader("Seed Traits + Cadence (v51)")
    traits_pos_file = st.file_uploader("Seed traits POSITIVE CSV (optional; autoloads if present)", type=["csv"], key="traits_pos_file")
    traits_neg_file = st.file_uploader("Seed traits NEGATIVE CSV (optional; autoloads if present)", type=["csv"], key="traits_neg_file")
    cadence_md_file = st.file_uploader("Cadence file (.md report OR .txt/.csv cadence table)", type=["md","txt","csv"], key="cadence_md_file")

    st.markdown("**Additional Seed‑Trait tables (optional)**")
    core_extra_pos_file = st.file_uploader("Extra CORE seed traits POSITIVE CSV (optional)", type=["csv"], key="core_extra_pos_file")
    core_extra_neg_file = st.file_uploader("Extra CORE seed traits NEGATIVE CSV (optional)", type=["csv"], key="core_extra_neg_file")
    member_traits_pos_file = st.file_uploader("Member seed traits POSITIVE CSV (optional)", type=["csv"], key="member_traits_pos_file")
    member_traits_neg_file = st.file_uploader("Member seed traits NEGATIVE CSV (optional)", type=["csv"], key="member_traits_neg_file")
    enable_member_seed_traits = st.checkbox("Enable Member Seed‑Trait overrides (soft, after MODE/LAST)", value=False, key="enable_member_seed_traits")
    member_seed_traits_weight = st.slider("Member Seed‑Trait weight", 0.0, 3.0, 0.25, 0.05, key="member_seed_traits_weight")

    enable_seed_traits = st.checkbox("Enable Seed Traits boost (soft only)", value=True, key="enable_seed_traits")
    seed_traits_weight = st.slider("Seed Traits weight", 0.0, 3.0, 0.35, 0.05, key="seed_traits_weight")
    enable_cadence = st.checkbox("Enable Cadence boost (soft only)", value=True, key="enable_cadence")
    cadence_weight = st.slider("Cadence weight", 0.0, 3.0, 0.25, 0.05, key="cadence_weight")

    # --- compatibility aliases (do not change UI; ensures downstream code sees expected keys)
    st.session_state["enable_seed_traits_boost"] = bool(enable_seed_traits)
    st.session_state["enable_cadence_boost"] = bool(enable_cadence)
    enable_seed_traits_boost = bool(enable_seed_traits)
    enable_cadence_boost = bool(enable_cadence)

    # Keep these weights conservative by default
    due_weight = st.slider("DuePressure weight", 0.0, 3.0, 0.20, 0.05, key="due_weight")
    pos_weight = st.slider("Position-percentile weight", 0.0, 3.0, 0.25, 0.05, key="pos_weight")

    map_file = map24_file  # backward-compatible alias

    exclude_md = st.checkbox("Exclude Maryland (MD)", value=True, help="Optional global exclusion. When enabled (default), MD rows are removed from both the baseline and 24h files before ranking.")
    st.session_state["exclude_md"] = exclude_md

    st.divider()
    st.subheader("Trigger Map (39-play list) — optional boost")
    _apply = st.checkbox("Apply Trigger Map boost", value=False)
    st.session_state["_apply_trigger_map"] = _apply
    _pts = st.slider("Trigger boost points", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    st.session_state["_trigger_boost_points"] = float(_pts)


    st.divider()
    st.divider()
    with st.expander("Build checklist (do not omit)", expanded=False):
        # --- Live status (auto) ---
        st.markdown("### Live status")
        _sel_now = st.session_state.get("cores_for_cache_ms", []) or st.session_state.get("selected_cores", []) or []
        st.write({
            "hardcoded_daily_doubles_cores": len(CORE_PRESETS),
            "selected_cores_now": len(_sel_now),
            "selected_cores_list": _sel_now[:25] + (["…"] if len(_sel_now) > 25 else []),
        })

        st.markdown("""
**A. Core + cache**
- **A1** Multi-core selection (core dropdown + multi-select)
- **A2** Cache Builder: build baseline cache for selected cores
- **A3** Show tabs for all selected cores (optional) in Core view
- **A4** Core ranking percentile map (tie-breaker) in Northern Lights view
- **A5** Bucket method: Top 12 + DueIndex 13–60 (8 picks)
- **A6** Straights module optional last (does not run unless enabled)

**B. Northern Star / Northern Lights**
- **B1** Northern Star (per-core) ranking view
- **B2** Northern Star buckets per core (Base + Due)
- **B3** Northern Lights master playlist (cross-core)
- **B4** Master playlist scoring is deterministic (stable tie-break)
- **B5** Optional Trigger Map boost (39-play list)

**C. Maps / percentiles**
- **C1** Global Northern Star position percentile map (1–78)
- **C2** Per-core percentile map tabs for selected cores

**D. Cadence & behavior (soft-only)**
- **D1** Cadence report integration (soft boost / transparency)
- **D2** Core-specific behavior tables (per-core stats cached)

**E. Self-maintenance**
- **E1** Local rolling ~3-year baseline store (append from 24h)
- **E2** Purge rows older than ~3 years (automatic)
- **E3** Store status panel: rows + date range + last updated
- **E4** One-click store rebuild/reset (safety)

If any item is missing in a build, treat it as a regression and restore it before adding new features.
""")


    st.subheader("Self-update rolling baseline (optional)")
    use_store = st.checkbox("Use local rolling ~3-year baseline store", value=False, help="Keeps a local rolling baseline by appending new rows from the 24h file and purging rows older than ~3 years. This improves speed and keeps your baseline fresh without you manually editing the all-states file.")
    st.session_state["use_store"] = use_store

    if use_store:
        store_df_preview = load_baseline_store()
        store_meta = _read_meta(BASELINE_STORE_BASE)
        store_rows = int(store_meta.get("rows", store_df_preview.shape[0] if store_df_preview is not None else 0) or 0)
        store_max = store_meta.get("max_date", "") or (str(store_df_preview["Date"].max()) if store_df_preview is not None and not store_df_preview.empty else "")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Initialize/overwrite store from uploaded all-states file"):
                if master_file is None:
                    st.warning("Upload the all-states history file first.")
                else:
                    try:
                        master_file.seek(0)
                    except Exception:
                        pass
                    try:
                        df_init = try_read_tablelike(master_file)
                        df_init = purge_to_rolling_3y(df_init, years=3)
                        ok, wrote = write_baseline_store(df_init, note="Initialized from uploaded all-states file (rolling 3y).")
                        st.success(f"Baseline store saved: {wrote}")
                    except Exception as e:
                        st.error(f"Could not initialize store: {e}")
                    try:
                        master_file.seek(0)
                    except Exception:
                        pass
        with colB:
            if st.button("Append 24h file into store (and purge)"):
                if map24_file is None:
                    st.warning("Upload the 24h file first.")
                else:
                    try:
                        map24_file.seek(0)
                    except Exception:
                        pass
                    try:
                        df_new = try_read_tablelike(map24_file)
                        df_store = load_baseline_store()
                        merged, added = append_from_24h(df_store, df_new)
                        merged = purge_to_rolling_3y(merged, years=3)
                        ok, wrote = write_baseline_store(merged, note=f"Appended from 24h file (+{added} new rows), then purged to rolling 3y.")
                        st.success(f"Updated store: +{added} new rows. Saved: {wrote}")
                    except Exception as e:
                        st.error(f"Could not append 24h into store: {e}")
# ---------- Core selection ----------
st.header("Core selection")

# Start from the curated preset list, then union with any cores detected from data (if present)
_detected_cores: list[str] = []
try:
    if isinstance(df_all, pd.DataFrame) and "Core" in df_all.columns:
        _det = df_all["Core"].dropna().astype(str).str.extract(r"(\d{1,3})", expand=False).dropna().unique().tolist()
        _det = [str(x).zfill(3) for x in _det]
        _det = [c for c in _det if c.isdigit()]
        _detected_cores = sorted(set(_det))
except Exception:
    _detected_cores = []

available_cores = sorted(set([str(c).zfill(3) for c in CORE_PRESETS] + _detected_cores))

# 7-core mode (default ON): limits core-selection & cache building to the 7 special cores.
mode_7core = st.checkbox(
    "7-core mode (limit app to 7 cores)",
    value=True,
    key="mode_7core",
    help="When ON, the app focuses on the 7 special cores (246,168,589,019,468,236,025) and defaults core selection to that set. Turn OFF to use all tracked cores."
)
if mode_7core:
    available_cores = [str(c).zfill(3) for c in SPECIAL_7_CORES]

cores = available_cores  # alias used throughout UI

# Ensure default core exists
default_core = str(getattr(cfg, "default_core", "389")).zfill(3)
if default_core not in available_cores:
    available_cores = [default_core] + available_cores

# Persist selection
if "cores_for_cache" not in st.session_state:
    st.session_state.cores_for_cache = [default_core]

# Core view dropdown (single core)
# View core (dropdown)
vc_default = str(default_core).zfill(3) if str(default_core).zfill(3) in cores else (str(cores[0]).zfill(3) if cores else '000')
vc = vc_default
try:
    vc = str(st.session_state.get('view_core', vc_default)).zfill(3)
except Exception:
    vc = vc_default
if vc not in cores:
    vc = vc_default
view_core = st.selectbox("View core (dropdown)", cores, index=cores.index(vc) if vc in cores else 0, key="view_core")
core_for_view = view_core  # backward-compatible variable name

# Multi-core selection for cache build / batch tools
# Keep a stable widget key, and initialize it only once to avoid Streamlit warnings.
if 'cores_for_cache_ms' not in st.session_state:
    if st.session_state.get('mode_7core', True):
        st.session_state.cores_for_cache_ms = [str(c).zfill(3) for c in SPECIAL_7_CORES]
    else:
        st.session_state.cores_for_cache_ms = list(st.session_state.get('cores_for_cache', [view_core]))
if view_core not in [str(c).zfill(3) for c in st.session_state.cores_for_cache_ms]:
    st.session_state.cores_for_cache_ms = [*st.session_state.cores_for_cache_ms, view_core]
_csel1, _csel2, _csel3 = st.columns([1,1,1])
with _csel1:
    if st.button("Select all cores", key="btn_select_all_cores"):
        st.session_state["cores_for_cache_ms"] = list(cores)
        _rerun()
with _csel2:
    if st.button("Clear selection", key="btn_clear_all_cores"):
        st.session_state["cores_for_cache_ms"] = []
        _rerun()
with _csel3:
    # Quick-pick for the 7-core bundle (no behavior changes elsewhere)
    if st.button("Select 7-core set", key="btn_select_7core_set", help="Quick-select the 7 special cores: 246, 168, 589, 019, 468, 236, 025"):
        st.session_state["cores_for_cache_ms"] = [str(c).zfill(3) for c in SPECIAL_7_CORES]
        _rerun()

cores_for_cache_ms = st.multiselect(
    "Cores to include for cache building / batch tools",
    cores,
    key="cores_for_cache_ms",
    help="Select one or more 3-digit cores. Cache building and batch tools will use this list.",
)
st.session_state.cores_for_cache = [str(c).zfill(3) for c in cores_for_cache_ms]

# --- SAFETY: always define cores_for_cache so later UI blocks can't NameError
# (Some sections reference cores_for_cache even if only cores_for_cache_ms exists.)
cores_for_cache = list(st.session_state.get('cores_for_cache') or st.session_state.get('cores_for_cache_ms') or st.session_state.get('selected_cores') or [])
cores_for_cache = [str(c).zfill(3) for c in cores_for_cache]
if not cores_for_cache:
    cores_for_cache = [str(getattr(cfg, 'default_core', '389')).zfill(3)]
st.header("Northern Star window")

window_days = st.radio("Window (days)", options=[180, 365], index=0, horizontal=True)
cfg = RankConfig(window_days=window_days)


st.divider()
st.header("Rare Engine trigger — AAAB + AABB")
r1 = st.checkbox("R1: Top‑20% baseline AAAB+AABB", value=True)
r2 = st.checkbox("R2: Top‑20% recent (last window)", value=True)
r3 = st.checkbox("R3: 24h has ≥3 AAAB/AABB hits across ≥3 streams", value=True)
r4 = st.checkbox("R4: 24h cluster ∈ Top‑10 positions", value=True)

st.divider()
st.header("Ultra‑Rare trigger — AAAA")
q1 = st.checkbox("Q1: Top‑10% quad baseline", value=True)
q2 = st.checkbox("Q2: Due pressure ≥ P90", value=True)
q3 = st.checkbox("Q3: 24h quad exists (core digits)", value=True)
q4 = st.checkbox("Q4: 24h cluster ∈ Top‑5 positions", value=True)

st.divider()
straights_opt = st.checkbox("Generate straights shortlist (optional last)", value=False)


# Load data
use_store = bool(st.session_state.get("use_store", False))

if use_store:
    # Prefer the on-disk rolling store if present
    df_all = load_baseline_store()
    # If the store is empty but the user uploaded a master file, auto-initialize (rolling 3y)
    if (df_all is None or df_all.empty) and master_file is not None:
        try:
            master_file.seek(0)
        except Exception:
            pass
        df_all = try_read_tablelike(master_file)
        df_all = purge_to_rolling_3y(df_all, years=3)
        try:
            write_baseline_store(df_all, note="Auto-initialized store from uploaded all-states file (rolling 3y).")
        except Exception:
            pass
        try:
            master_file.seek(0)
        except Exception:
            pass
else:
    if master_file is not None:
        # Ensure we read from start on every rerun (Streamlit UploadedFile pointer can be at EOF)
        try:
            master_file.seek(0)
        except Exception:
            pass
        df_all = try_read_tablelike(master_file)
        # Reset pointer for any later consumers
        try:
            master_file.seek(0)
        except Exception:
            pass
    else:
        df_all = pd.DataFrame()

prev_picklist = pd.DataFrame()
df_24h = pd.DataFrame()
if map24_file:
    try:
        try:
            map24_file.seek(0)
        except Exception:
            pass
        df_24h = try_read_tablelike(map24_file)
        try:
            map24_file.seek(0)
        except Exception:
            pass
    except Exception:
        # Many users upload a simple pick-list here (one 4-digit number per line).
        # Accept it without crashing; it will NOT be used for 24h engines or baseline self-update.
        try:
            map24_file.seek(0)
        except Exception:
            pass
        try:
            prev_picklist = try_read_picklist(map24_file)
            if not prev_picklist.empty:
                st.info(
                    "Optional 24h/previous-day file detected as a pick-list (not a LotteryPost history export). "
                    "It will be used only for annotation/downranking where applicable."
                )
        except Exception as e:
            st.warning(f"Could not parse optional 24h/previous-day file: {e}")


if exclude_md and not df_all.empty:
    df_all = df_all[df_all["State"].astype(str).str.strip().str.lower() != "maryland"].copy()
if exclude_md and not df_24h.empty:
    df_24h = df_24h[df_24h["State"].astype(str).str.strip().str.lower() != "maryland"].copy()

# Back-compat alias used by the Northern Lights block
df_24 = df_24h


# Auto-clear cached percentile maps when input data changes
if use_store:
    _m = _read_meta(BASELINE_STORE_BASE)
    all_hash = f"store|{_m.get('max_date','')}|{_m.get('rows','')}"
else:
    all_hash = file_fingerprint(master_file)
map_hash = file_fingerprint(map24_file)
if "data_hash_all" not in st.session_state:
    st.session_state["data_hash_all"] = ""
if "data_hash_24h" not in st.session_state:
    st.session_state["data_hash_24h"] = ""

if all_hash and all_hash != st.session_state["data_hash_all"]:
    st.session_state["pos_map_cache"] = {}
    st.session_state["data_hash_all"] = all_hash
    st.session_state["recompute_token"] = ""  # will refresh on next compute
if map_hash != st.session_state["data_hash_24h"]:
    st.session_state["pos_map_cache"] = {}
    st.session_state["data_hash_24h"] = map_hash
    st.session_state["recompute_token"] = ""

# Show data freshness (so the instructions never need updating)
last_all = most_recent_date(df_all)
last_24 = most_recent_date(df_24h)


# ---- v51: Seed Traits + Cadence data (autoload + optional uploads)
DEFAULT_TRAITS_POS_PATH = "family_seed_traits_DOUBLES_top_positive_EXPANDED.csv"
DEFAULT_TRAITS_NEG_PATH = "family_seed_traits_DOUBLES_top_negative_EXPANDED.csv"
DEFAULT_CADENCE_MD_PATH = "family_cadence_report.md"

# Optional extra trait tables (not required)
DEFAULT_CORE_EXTRA_POS_PATH = "family_seed_traits_DOUBLES_core_extra_positive.csv"
DEFAULT_CORE_EXTRA_NEG_PATH = "family_seed_traits_DOUBLES_core_extra_negative.csv"
DEFAULT_MEMBER_TRAITS_POS_PATH = "family_seed_traits_DOUBLES_member_extra_positive.csv"
DEFAULT_MEMBER_TRAITS_NEG_PATH = "family_seed_traits_DOUBLES_member_extra_negative.csv"

# Load traits (uploaded overrides local if provided)
seed_traits_pos_df = _read_local_or_uploaded_csv(globals().get("traits_pos_file", None), DEFAULT_TRAITS_POS_PATH)
seed_traits_neg_df = _read_local_or_uploaded_csv(globals().get("traits_neg_file", None), DEFAULT_TRAITS_NEG_PATH)
seed_traits_pos_lookup = _build_traits_lookup(seed_traits_pos_df)
seed_traits_neg_lookup = _build_traits_lookup(seed_traits_neg_df)

# Extra CORE traits (optional)
core_extra_pos_df = _read_local_or_uploaded_csv(globals().get("core_extra_pos_file", None), DEFAULT_CORE_EXTRA_POS_PATH)
core_extra_neg_df = _read_local_or_uploaded_csv(globals().get("core_extra_neg_file", None), DEFAULT_CORE_EXTRA_NEG_PATH)
core_extra_pos_lookup = _build_traits_lookup(core_extra_pos_df)
core_extra_neg_lookup = _build_traits_lookup(core_extra_neg_df)

# Combined lookups: base + optional extra core tables (extras may be empty)
seed_traits_pos_lookup = {**seed_traits_pos_lookup, **(core_extra_pos_lookup or {})}
seed_traits_neg_lookup = {**seed_traits_neg_lookup, **(core_extra_neg_lookup or {})}

# Audit panel expects explicit core-level lookup names. In this app, the
# "seed traits" tables are the core-level traits, so alias them here.
core_traits_pos_lookup = seed_traits_pos_lookup
core_traits_neg_lookup = seed_traits_neg_lookup

# Member traits (optional)
member_traits_pos_df = _read_local_or_uploaded_csv(globals().get("member_traits_pos_file", None), DEFAULT_MEMBER_TRAITS_POS_PATH)
member_traits_neg_df = _read_local_or_uploaded_csv(globals().get("member_traits_neg_file", None), DEFAULT_MEMBER_TRAITS_NEG_PATH)
member_traits_pos_lookup = _build_traits_lookup(member_traits_pos_df)
member_traits_neg_lookup = _build_traits_lookup(member_traits_neg_df)
member_traits_pos_lut = _build_member_traits_lookup(member_traits_pos_df)
member_traits_neg_lut = _build_member_traits_lookup(member_traits_neg_df)

cadence_report_text = _read_local_or_uploaded_text(globals().get("cadence_md_file", None), DEFAULT_CADENCE_MD_PATH)
cadence_table_df = _parse_cadence_table_from_text(cadence_report_text)

# Precompute per-stream seed + last5 union digits (for Seed Traits feature)
_prev_seed_by_stream: Dict[str, str] = {}
_last5_union_by_stream: Dict[str, set] = {}

try:
    # Determine the most recent 4-digit seed per stream (prefer 24h map if present)
    if df_24h is not None and not df_24h.empty and "Stream" in df_24h.columns and "Result" in df_24h.columns:
        _tmp = df_24h.copy()
        if "Date" in _tmp.columns:
            _tmp = _tmp.sort_values("Date")
        # take last per stream
        _prev = _tmp.groupby("Stream", as_index=False).tail(1)
        _prev_seed_by_stream = dict(zip(_prev["Stream"].astype(str), _prev["Result"].astype(str)))
    if not _prev_seed_by_stream:
        _tmp = df_all.copy()
        _tmp = _tmp.sort_values("Date")
        _prev = _tmp.groupby("Stream", as_index=False).tail(1)
        _prev_seed_by_stream = dict(zip(_prev["Stream"].astype(str), _prev["Result"].astype(str)))

    # last5 union digits per stream from df_all (most recent 5 rows)
    _tmp = df_all.sort_values("Date")
    for s, g in _tmp.groupby("Stream"):
        tail = g.tail(5)
        digs = set("".join(tail["Result"].astype(str).tolist()))
        _last5_union_by_stream[str(s)] = digs
except Exception:
    pass

today = datetime.date.today()


def _render_backtest_walk_forward(df_all: pd.DataFrame, cfg: "RankConfig", cores_for_cache: list[str]) -> None:
    def _row_get(_row, _key, _default=""):
        """Safe column access for Series / dict / itertuples namedtuple rows."""
        if _row is None:
            return _default
        try:
            if hasattr(_row, "get") and callable(getattr(_row, "get")):
                try:
                    _v = _row.get(_key, _default)
                    return _default if _v is None else _v
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(_row, "_asdict"):
                _d = _row._asdict()
                _v = _d.get(_key, _default)
                return _default if _v is None else _v
        except Exception:
            pass
        try:
            if hasattr(_row, _key):
                _v = getattr(_row, _key)
                return _default if _v is None else _v
        except Exception:
            pass
        return _default
    """Walk-forward backtest (no future leakage).

    - For each test_date:
      - train_df = rows with Date < test_date
      - build per-core stream ranking/buckets from train_df ONLY
      - score winner rows on test_date against those buckets
    """
    if df_all is None or df_all.empty:
        st.warning("No data loaded.")
        return
    if "Date" not in df_all.columns or "Stream" not in df_all.columns or "Result" not in df_all.columns:
        st.error("Backtest requires columns: Date, Stream, Result.")
        return

    # Pick cores
    all_cores = list(CORE_PRESETS)
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        use_all = st.checkbox("Test ALL cores", value=False, key="bt_use_all_cores")
    with c2:
        include_rare = st.checkbox("Include AAAB/AABB/AAAA members", value=False, key="bt_include_rare")
    with c3:
        max_dates = st.number_input("Max test dates", min_value=1, max_value=3650, value=120, step=10, key="bt_max_dates")


    st.markdown("##### Member‑pick tracking (optional)")
    m1, m2 = st.columns([1.1, 1.9])
    with m1:
        member_track = st.checkbox("Track member accuracy (Top1/Top2)", value=True, key="bt_member_track")
    with m2:
        member_basis_label = st.selectbox(
            "Member predictor basis",
            ["Per‑core (all streams)", "Per‑core + stream"],
            index=0,
            key="bt_member_basis",
            help="Per‑core uses all streams to learn which member is most common for that core. Per‑core+stream learns separately per stream (more specific, but fewer samples).",
        )
    member_basis = "core_stream" if member_basis_label.startswith("Per‑core + stream") else "core"

    if use_all:
        cores_sel = all_cores
    else:
        default_cores = cores_for_cache if cores_for_cache else ["389"]
        cores_sel = st.multiselect(
            "Cores to test",
            options=all_cores,
            default=[c for c in default_cores if c in all_cores] or [all_cores[0]],
            key="bt_cores_sel",
        )

    if not cores_sel:
        st.info("Select at least one core to backtest.")
        return

    # Date range
    dmin = _parse_date_series(df_all["Date"]).min()
    dmax = _parse_date_series(df_all["Date"]).max()
    if pd.isna(dmin) or pd.isna(dmax):
        st.error("Could not parse Date values for backtest.")
        return

    default_start = (dmax - pd.Timedelta(days=90)).date() if (dmax - dmin).days > 120 else dmin.date()
    start_date, end_date = st.date_input(
        "Test date range (inclusive)",
        value=(default_start, dmax.date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
        key="bt_date_range",
    )

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    only_hit_days = st.checkbox("Diagnostic only: evaluate only days where a selected core member hit (faster) — NOT valid for EV", value=True, key="bt_only_hit_days")

    run = st.button("Run walk-forward backtest", key="bt_run_btn")

    # Settings signature: if sliders/toggles change, we must not treat old results as current.
    # 24h map is optional; treat it as enabled only when an upload exists and parsed rows are present.
    use_24h_map = bool(map24_file is not None and isinstance(df_24, pd.DataFrame) and (not df_24.empty))
    current_sig = _bt_settings_sig(
        build=APP_BUILD_ID,
        exclude_md=bool(exclude_md),
        window_days=int(cfg.window_days),
        member_track=bool(member_track),
        member_basis=str(member_basis),
        enable_member_seed_traits=bool(enable_member_seed_traits),
        member_seed_traits_weight=float(member_seed_traits_weight or 0.0),
        enable_seed_traits=bool(enable_seed_traits),
        seed_traits_weight=float(seed_traits_weight or 0.0),
        enable_cadence=bool(enable_cadence),
        cadence_weight=float(cadence_weight or 0.0),
        due_weight=float(due_weight or 0.0),
        pos_weight=float(pos_weight or 0.0),
        cores=[str(c) for c in cores_sel],
        include_rare=bool(include_rare),
        use_24h_map=bool(use_24h_map),
        member_pos_rules=int(len(member_traits_pos_lut) if isinstance(member_traits_pos_lut, dict) else 0),
        member_neg_rules=int(len(member_traits_neg_lut) if isinstance(member_traits_neg_lut, dict) else 0),
    )

    use_cached = False
    if not run:
        # After Streamlit reruns, the button is False; keep the last results visible
        _cached_out = st.session_state.get("wf_backtest_out", None)
        _cached_sig = st.session_state.get("wf_backtest_sig", None)
        if _cached_out is not None and hasattr(_cached_out, "empty") and (not _cached_out.empty) and (_cached_sig == current_sig):
            use_cached = True
            out = _cached_out.copy()
            st.info("Showing the most recent backtest results (from cache). Click **Run walk-forward backtest** to recompute.")
        elif _cached_out is not None and hasattr(_cached_out, "empty") and (not _cached_out.empty) and (_cached_sig != current_sig):
            st.warning("Cached backtest results were generated with different settings (weights/toggles/files). Click **Run walk-forward backtest** to recompute for the current settings.")
        else:
            st.info("Click **Run walk-forward backtest** to generate results.")
            return

    if not use_cached:
        df = df_all.copy()
        df = df[df["Date"].notna()]
        df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
        if df.empty:
            st.warning("No rows in the selected date range.")
            return

        # Build member->core reverse index
        member_to_cores: dict[str, list[str]] = {}
        for core in cores_sel:
            members = []
            # Always include the core's main family members (AABC/ABBC/ABCC)
            members.extend(members_from_core(core, "AABC"))
            # Optionally include higher-rarity structures for the same core
            if include_rare:
                members.extend(members_from_core(core, "AAAB"))
                members.extend(members_from_core(core, "AABB"))
                members.extend(members_from_core(core, "AAAA"))
            for mem in members:
                member_to_cores.setdefault(box_key(mem), []).append(core)

        # Winners by date
        winners_by_date = {d: g for d, g in df.groupby(df["Date"].dt.normalize())}

        # Determine dates to evaluate
        all_dates_sorted = sorted(winners_by_date.keys())
        if only_hit_days:
            candidate_dates = []
            for d in all_dates_sorted:
                g = winners_by_date[d]
                hit = False
                for w in g["Result"].astype(str).tolist():
                    if member_to_cores.get(box_key(w.strip()), None):
                        hit = True
                        break
                if hit:
                    candidate_dates.append(d)
            dates_to_test = candidate_dates
        else:
            dates_to_test = all_dates_sorted

        if not dates_to_test:
            st.warning("No hit days found for the selected cores in this date range.")
            return

        if len(dates_to_test) > int(max_dates):
            dates_to_test = dates_to_test[-int(max_dates):]

        # Cache per (core, as_of_date, window_days) within this run
        per_core_cache: dict[tuple[str, pd.Timestamp, int], pd.DataFrame] = {}
        per_core_buckets: dict[tuple[str, pd.Timestamp, int], dict] = {}

        # Member-pick prediction cache for this run: (core, test_date, window_days, basis, stream) -> dict
        member_pred_cache: dict[tuple[str, pd.Timestamp, int, str, str | None], dict[str, Any]] = {}

        rows = []
        # Speed: avoid re-scanning the full dataframe for each test_date.
        # Sort once, then slice by index using searchsorted (equivalent to df_all[df_all["Date"] < test_date]).
        _df_sorted = df_all.sort_values("Date").reset_index(drop=True)
        _date_vals = _parse_date_series(_df_sorted['Date']).to_numpy(dtype='datetime64[ns]')

        for test_date in dates_to_test:
            _td = np.datetime64(pd.to_datetime(test_date, errors='coerce').normalize())
            if np.isnat(_td):
                continue
            cutoff = int(np.searchsorted(_date_vals, _td, side='left'))
            if cutoff <= 0:
                continue
            train_df = _df_sorted.iloc[:cutoff]
            day_winners = winners_by_date.get(test_date)
            if day_winners is None or day_winners.empty:
                continue

            for _, wr in day_winners.iterrows():
                stream = str(_row_get(wr, "Stream", "")).strip()
                winner = str(_row_get(wr, "Result", "")).strip()
                wk = box_key(winner)
                hit_cores = member_to_cores.get(wk, [])
                if not hit_cores:
                    continue

                for core in hit_cores:
                    key = (core, test_date, int(cfg.window_days))
                    if key not in per_core_cache:
                        stats_df = compute_stream_stats(train_df, core, window_days=int(cfg.window_days))
                        per_core_cache[key] = stats_df
                        per_core_buckets[key] = bucket_recommendations(stats_df, cfg)

                    stats_df = per_core_cache[key]
                    buckets = per_core_buckets[key]
                    base_streams = set(buckets["Top12BaseScore"]["Stream"].astype(str).tolist()) if "Top12BaseScore" in buckets else set()
                    due_streams = set(buckets["Due8"]["Stream"].astype(str).tolist()) if "Due8" in buckets else set()
                    predicted = stream in base_streams or stream in due_streams
                    bucket = "Both" if (stream in base_streams and stream in due_streams) else ("BaseScore" if stream in base_streams else ("Due8" if stream in due_streams else "None"))

                    # Pull per-stream stats if present
                    stat_row = None
                    try:
                        stat_row = stats_df.loc[stats_df["Stream"].astype(str) == stream].iloc[0]
                    except Exception:
                        stat_row = None



                    # Member labels + walk-forward member-pick prediction (family only: AABC/ABBC/ABCC)

                    actual_member_label = core_member_label(core, winner, include_rare=bool(include_rare)) if member_track else None

                    actual_family_member = actual_member_label if actual_member_label in ("AABC", "ABBC", "ABCC") else None


                    pred_member_top1 = None

                    pred_member_top2 = None

                    pred_member_n = 0

                    train_cnt_aabc = 0

                    train_cnt_abbc = 0

                    train_cnt_abcc = 0

                    member_hit_top1 = None

                    member_hit_top2 = None


                    if member_track:

                        mk = (

                            str(core),

                            pd.to_datetime(test_date).normalize(),

                            int(cfg.window_days),

                            str(member_basis),

                            (str(stream) if member_basis == "core_stream" else None),

                        )

                        if mk not in member_pred_cache:

                            member_pred_cache[mk] = predict_core_member(

                                df_all,

                                core,

                                pd.to_datetime(test_date).normalize(),

                                window_days=int(cfg.window_days),

                                basis=str(member_basis),

                                stream=(str(stream) if member_basis == "core_stream" else None),

                                include_rare=False,  # compare only AABC/ABBC/ABCC

                            )

                        mp = member_pred_cache.get(mk, {})
                        pred_member_top1_mode = mp.get("top1")
                        pred_member_top2_mode = mp.get("top2")
                        pred_member_top1 = pred_member_top1_mode
                        pred_member_top2 = pred_member_top2_mode

                        # Apply member-level seed-trait overrides (soft) if enabled + tables provided
                        if enable_member_seed_traits and (member_traits_pos_lut or member_traits_neg_lut):
                            try:
                                variants = _member_prediction_variants(
                                    df_all=df_all,
                                    traits_pos_df=traits_pos_df,
                                    core_key=str(core),
                                    test_date=pd.to_datetime(test_date).normalize(),
                                    window_days=int(cfg.window_days),
                                    stream=str(stream) if member_basis == "core_stream" else str(stream),
                                    basis="core_stream" if member_basis == "core_stream" else "all-streams",
                                    min_stream_hits_for_last=3,
                                    enable_member_seed_traits_overrides=True,
                                    member_seed_traits_weight=float(member_seed_traits_weight or 0.25),
                                    member_traits_pos_lut=member_traits_pos_lut,
                                    member_traits_neg_lut=member_traits_neg_lut,
                                )
                                if variants.get("MEMBER_TRAIT_OVERRIDE"):
                                    pred_member_top1 = variants.get("MEMBER_TRAIT_OVERRIDE")
                                    # use the second-best adjusted pick when available
                                    pred_member_top2 = variants.get("MEMBER_TRAIT_TOP2") or pred_member_top2_mode
                            except Exception:
                                pass

                        pred_member_n = int(mp.get("n") or 0)

                        cnts = mp.get("counts") or {}

                        train_cnt_aabc = int(cnts.get("AABC") or 0)

                        train_cnt_abbc = int(cnts.get("ABBC") or 0)

                        train_cnt_abcc = int(cnts.get("ABCC") or 0)


                        if actual_family_member and pred_member_top1:

                            member_hit_top1 = (actual_family_member == pred_member_top1)

                            member_hit_top2 = (actual_family_member == pred_member_top1) or (pred_member_top2 is not None and actual_family_member == pred_member_top2)

                    rows.append({
                        "Date": test_date.date(),
                        "Stream": stream,
                        "Winner": winner,
                        "Core": core,
                        "Predicted": bool(predicted),
                        "Bucket": bucket,
                        "RankPos": (int(stat_row["RankPos"]) if (stat_row is not None and "RankPos" in stat_row and pd.notna(stat_row["RankPos"])) else None),
                        "BaseScoreRank": (int(stat_row["BaseScoreRank"]) if (stat_row is not None and "BaseScoreRank" in stat_row and pd.notna(stat_row["BaseScoreRank"])) else None),
                        "HitsWindow": (int(stat_row["HitsWindow"]) if (stat_row is not None and "HitsWindow" in stat_row and pd.notna(stat_row["HitsWindow"])) else None),
                        "DaysSinceLastHit": (int(stat_row["DaysSinceLastHit"]) if (stat_row is not None and "DaysSinceLastHit" in stat_row and pd.notna(stat_row["DaysSinceLastHit"])) else None),
                        "AsOfMaxDate": pd.to_datetime(train_df["Date"], errors="coerce").max().date() if "Date" in train_df.columns else None,
                        "ActualMemberLabel": actual_member_label,
                        "ActualFamilyMember": actual_family_member,
                        "PredMemberTop1_Mode": pred_member_top1_mode,
                        "PredMemberTop2_Mode": pred_member_top2_mode,
                        "PredMemberTop1": pred_member_top1,
                        "PredMemberTop2": pred_member_top2,
                        "MemberTraitOverrideApplied": bool(enable_member_seed_traits and (member_traits_pos_lut or member_traits_neg_lut) and (pred_member_top1 != pred_member_top1_mode or (pred_member_top2 is not None and pred_member_top2 != pred_member_top2_mode))),
                        "MemberHitTop1": member_hit_top1,
                        "MemberHitTop2": member_hit_top2,
                        "MemberTrainN": pred_member_n,
                        "TrainCnt_AABC": train_cnt_aabc,
                        "TrainCnt_ABBC": train_cnt_abbc,
                        "TrainCnt_ABCC": train_cnt_abcc,
                    })

        if not rows:
            st.warning("No matching core-family wins found in the evaluated dates.")
            return

        out = pd.DataFrame(rows).sort_values(["Date", "Core", "Predicted"], ascending=[True, True, False])
        # Persist latest backtest output for other panels (e.g., Auto Profit Planner)
        st.session_state["wf_backtest_out"] = out.copy()
        st.session_state["wf_backtest_sig"] = current_sig


    # Summary
    total = len(out)
    hits = int(out["Predicted"].sum())
    st.success(f"Evaluated {total} core-family wins; predicted {hits} ({(hits/total*100):.1f}%).")


    # Optional: member-pick accuracy (Top1/Top2) for family members (AABC/ABBC/ABCC)
    if member_track and (not out.empty):
        st.markdown("#### Member pick accuracy (Top1/Top2)")
        st.caption("These stats answer: when a core hit, was the *predicted* family member the *actual* family member? (Top1 = exact pick; Top2 = in top-2 picks).")
        need_cols = ["ActualFamilyMember","PredMemberTop1","PredMemberTop2","MemberHitTop1","MemberHitTop2","MemberTrainN"]
        missing = [c for c in need_cols if c not in out.columns]
        if missing:
            st.warning(f"Member columns missing from output: {missing}. (This should not happen; please re-run.)")
        else:
            member_df = out.dropna(subset=["ActualFamilyMember"]).copy()
            # Ensure these columns are truly numeric for reliable aggregation (Streamlit/Pandas can infer object when nulls exist)
            for _col in ("MemberHitTop1", "MemberHitTop2", "MemberHitTopK"):
                if _col in member_df.columns:
                    member_df[_col] = member_df[_col].fillna(False).astype(int)
            for _col in ("UniversalScore", "RecomputedScore", "BaseTerm", "DueTerm", "PosTerm", "SeedTerm", "CadTerm", "CadenceScore", "PctStrength", "HitsPerWeek"):
                if _col in member_df.columns:
                    member_df[_col] = pd.to_numeric(member_df[_col], errors="coerce")

            if member_df.empty:
                st.info("No family-member hits in this test window (nothing to score for member accuracy).")
            else:
                agg = member_df.groupby("Core", dropna=False).agg(
                    N=("Core","size"),
                    Top1Hit=("MemberHitTop1","sum"),
                    Top2Hit=("MemberHitTop2","sum"),
                    AvgTrainN=("MemberTrainN","mean"),
                    MedTrainN=("MemberTrainN","median"),
                ).reset_index()
                agg["Top1Rate"] = (agg["Top1Hit"] / agg["N"]).round(4)
                agg["Top2Rate"] = (agg["Top2Hit"] / agg["N"]).round(4)
                st.dataframe(agg.sort_values(["Top2Rate","Top1Rate","N"], ascending=False), use_container_width=True)
                st.caption("Tip: if Top2Rate is strong but Top1Rate is weak, treat this as a *top-2 member shortlist* (play 2 members, not 1).")

    # Trust check: walk-forward (no leakage)
    leak_ok = True
    if (not out.empty) and ("AsOfMaxDate" in out.columns):
        try:
            _max_train = pd.to_datetime(out["AsOfMaxDate"], errors="coerce")
            _test = pd.to_datetime(out["Date"], errors="coerce")
            leak_ok = bool((_max_train <= (_test - pd.Timedelta(days=1))).fillna(True).all())
        except Exception:
            leak_ok = True
    st.caption("Leakage check: " + ("✅ OK" if leak_ok else "❌ FAILED") + " — AsOfMaxDate should be <= test_date-1 for all rows.")

    # -------------------------
    # Box-first / Straight-add-on KPIs (matches your real plan)
    # -------------------------
    if use_cached and (st.session_state.get("bt_last_kpi_sig", None) == current_sig):
        _kpi_df = st.session_state.get("bt_last_kpi_df", None)
        _tot = st.session_state.get("bt_last_kpi_totals", None)
        if _kpi_df is not None and hasattr(_kpi_df, "empty") and (not _kpi_df.empty):
            st.markdown("### Box-first KPI (matches your real plan)")
            st.caption("Showing KPIs from the most recent backtest run (cached).")
            try:
                total_draws_eval = int((_tot or {}).get("total_draws_eval", len(_kpi_df)))
                total_box_hits = int((_tot or {}).get("total_box_hits", int(_kpi_df["BoxHit"].sum()) if "BoxHit" in _kpi_df.columns else 0))
                total_box_tickets = int((_tot or {}).get("total_box_tickets", int(_kpi_df["BoxesPlayed"].sum()) if "BoxesPlayed" in _kpi_df.columns else 0))
                total_straight_hits = int((_tot or {}).get("total_straight_hits", int(_kpi_df["StraightHit"].sum()) if "StraightHit" in _kpi_df.columns else 0))
                total_straight_tickets = int((_tot or {}).get("total_straight_tickets", int(_kpi_df["StraightsPlayed"].sum()) if "StraightsPlayed" in _kpi_df.columns else 0))
            except Exception:
                total_draws_eval = len(_kpi_df)
                total_box_hits = 0
                total_box_tickets = 0
                total_straight_hits = 0
                total_straight_tickets = 0
            box_hit_rate = (total_box_hits / total_draws_eval) if total_draws_eval else 0.0
            straight_hit_rate = (total_straight_hits / total_straight_tickets) if total_straight_tickets else 0.0

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Total draws evaluated", f"{total_draws_eval}")
            t2.metric("Box hits", f"{total_box_hits} ({box_hit_rate*100:.2f}%)")
            t3.metric("Straight tickets", f"{total_straight_tickets}")
            t4.metric("Straight hits", f"{total_straight_hits} ({straight_hit_rate*100:.2f}%)" if total_straight_tickets else "0 (n/a)")

            try:
                _cpp = float(0.25)
            except Exception:
                _cpp = 0.25
            total_cost = _cpp * (total_box_tickets + total_straight_tickets)
            st.markdown("#### Ticket counts + cost (at $0.25 per ticket)")
            st.write({
                "TotalBoxTickets": int(total_box_tickets),
                "TotalStraightTickets": int(total_straight_tickets),
                "TotalCost($)": round(float(total_cost), 2),
            })
            st.markdown("#### Per-draw results (first 300 rows shown)")
            st.dataframe(_kpi_df.head(300), use_container_width=True, hide_index=True)
            st.download_button(
                "Download KPI per-draw results (CSV)",
                data=_kpi_df.to_csv(index=False).encode("utf-8"),
                file_name="backtest_box_straight_kpis.csv",
                mime="text/csv",
                key="bt_dl_box_straight_kpis_cached",
            )
        else:
            st.markdown("### Box-first KPI (matches your real plan)")
            st.info("Run walk-forward backtest to generate Box/Straight KPIs.")
    else:
        st.markdown("### Box-first KPI (matches your real plan)")
        st.caption(
            "These metrics evaluate **ALL draws** in the selected date range (no future leakage). "
            "They ignore the diagnostic 'only hit days' toggle so the denominator stays valid for EV/performance."
        )

        # Box plan definition: per selected core, play Top-2 members by default (AABC/ABBC),
        # except cores 246 and 168 which play ALL 3 (AABC/ABBC/ABCC) to match the Play Today rule.
        # Optionally include rare/ultra structures if enabled.
        kpi1, kpi2 = st.columns([1.2, 1.8])
        with kpi1:
            kpi_eval_straights = st.checkbox(
                "Also evaluate Straights add-on (walk-forward)",
                value=True,
                key="bt_kpi_eval_straights",
            )
        with kpi2:
            st.caption(
                "Diagnostic note: the 'Evaluate only hit days' checkbox is useful for member-pick analysis speed, "
                "but it changes the denominator. These KPIs always evaluate ALL draws."
            )

        # Straight settings (kept simple, conservative defaults; uses your real history only)
        sc1, sc2, sc3, sc4 = st.columns([1.0, 1.0, 1.0, 1.0])
        with sc1:
            kpi_straight_cap = st.number_input(
                "Straight tickets per stream/day (cap)",
                min_value=0,
                max_value=25,
                value=2,
                step=1,
                key="bt_kpi_straight_cap",
                help="Maximum number of straight tickets the optimizer can select per stream per day (0 disables straights).",
            )
        with sc2:
            kpi_topk_per_member = st.number_input(
                "Top-K straights per box (from history)",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key="bt_kpi_topk_per_member",
            )
        with sc3:
            kpi_min_stream_samples = st.number_input(
                "Min stream samples",
                min_value=1,
                max_value=5000,
                value=20,
                step=5,
                key="bt_kpi_min_stream_samples",
            )
        with sc4:
            kpi_min_topk_cov = st.number_input(
                "Min Top-K coverage",
                min_value=0.0,
                max_value=2.0,
                value=0.04,
                step=0.01,
                key="bt_kpi_min_topk_cov",
                help="Skip a box's straight optimizer if the cumulative probability across its Top-K straights is below this threshold.",
            )

        # Precompute the constant box set you would play (per core rules)
        def _boxes_to_play_for_core(_core: str) -> list[str]:
            _core = str(_core).zfill(3)
            base_members = members_from_core(_core, "AABC")  # [AABC, ABBC, ABCC]
            if not base_members:
                return []
            all3 = _core in {"246", "168"}
            played = base_members if all3 else base_members[:2]
            if include_rare:
                # Rare/Ultra members are additive (never remove standard members)
                try:
                    for _struct in ("AAAB", "AABB", "AAAA"):
                        _more = members_from_core(_core, _struct)
                        if _more:
                            played.extend(list(_more))
                except Exception:
                    pass
            # de-dupe while preserving order
            seen = set()
            out_played = []
            for b in played:
                b = str(b).strip()
                if not b or b in seen:
                    continue
                seen.add(b)
                out_played.append(b)
            return out_played

        played_boxes = []
        for _c in cores_sel:
            played_boxes.extend(_boxes_to_play_for_core(_c))
        played_boxes = [b for b in played_boxes if isinstance(b, str) and len(b.strip()) == 4 and b.strip().isdigit()]
        played_box_keys = sorted({box_key(b) for b in played_boxes})
        boxes_per_draw = len(played_box_keys)

        # Evaluate ALL dates in-range (cap by Max test dates) to keep denominator valid
        dates_all = list(all_dates_sorted)
        if len(dates_all) > int(max_dates):
            dates_all = dates_all[-int(max_dates):]

        kpi_rows = []
        total_draws_eval = 0
        total_box_hits = 0
        total_straight_hits = 0
        total_box_tickets = 0
        total_straight_tickets = 0

        # Ensure df_all Date is datetime for training
        _df_all_dt = df_all.copy()
        try:
            _df_all_dt["Date"] = _parse_date_series(_df_all_dt["Date"])
        except Exception:
            pass

        # --- SPEED: pre-sort once and use searchsorted slices (avoids O(N*D) boolean masks) ---
        _df_all_sorted = _df_all_dt.sort_values("Date").reset_index(drop=True)
        # Precompute box keys once for the entire sorted history (used by KPI straight add-on).
        if "__BOXKEY__" not in _df_all_sorted.columns:
            try:
                _df_all_sorted["__BOXKEY__"] = _df_all_sorted["Result"].astype(str).map(box_key)
            except Exception:
                pass
        _all_norm_arr = _df_all_sorted["Date"].dt.normalize().values.astype("datetime64[ns]")

        for _d in dates_all:
            g = winners_by_date.get(_d, None)
            if g is None or g.empty:
                continue

            # walk-forward training history: ALL rows strictly before this date (no leakage)
            _d64 = np.datetime64(pd.Timestamp(_d).normalize())
            _cut = int(np.searchsorted(_all_norm_arr, _d64, side="left"))
            hist_train = _df_all_sorted.iloc[:_cut]

            # Precompute straights per stream ONCE for this date (major speed-up).
            # NOTE: This path was the primary cause of "1 chart then runs forever" because it
            #       re-scanned the full training history for every (stream x member-box) pair.
            #       We instead build compact groupby counts once per date, restricted to the
            #       boxes you actually play, and then read top-k straights from those counts.
            selected_straights_by_stream = {}
            if kpi_eval_straights and int(kpi_straight_cap) > 0 and (hist_train is not None) and (not hist_train.empty):
                try:
                    _streams_today = [str(s).strip() for s in g["Stream"].astype(str).unique().tolist()] if "Stream" in g.columns else []
                except Exception:
                    _streams_today = []

                try:
                    _played_set = set([str(x) for x in played_boxes])
                except Exception:
                    _played_set = set()

                # Ensure __BOXKEY__ exists (should be precomputed globally, but guard anyway)
                if "__BOXKEY__" not in hist_train.columns:
                    try:
                        _ht = hist_train.copy()
                        _ht["__BOXKEY__"] = _ht["Result"].astype(str).map(box_key)
                        hist_train = _ht
                    except Exception:
                        pass

                # Build compact counts only for played boxes (huge speed-up)
                try:
                    ht = hist_train[hist_train["__BOXKEY__"].astype(str).isin(_played_set)][["Stream","__BOXKEY__","Result"]].copy()
                except Exception:
                    ht = pd.DataFrame()

                if ht is not None and (not ht.empty):
                    try:
                        ht["Stream"] = ht["Stream"].astype(str)
                        ht["__BOXKEY__"] = ht["__BOXKEY__"].astype(str)
                        ht["Result"] = ht["Result"].astype(str)
                    except Exception:
                        pass

                    # Stream-specific counts: (Stream, Box, Straight) -> cnt, plus n per (Stream, Box)
                    try:
                        sc = ht.groupby(["Stream","__BOXKEY__","Result"]).size().rename("cnt").reset_index()
                        sb_tot = sc.groupby(["Stream","__BOXKEY__"])["cnt"].sum().rename("n").reset_index()
                        sc = sc.merge(sb_tot, on=["Stream","__BOXKEY__"], how="left")
                    except Exception:
                        sc = pd.DataFrame(columns=["Stream","__BOXKEY__","Result","cnt","n"])

                    # Global counts (fallback): (Box, Straight) -> cnt, plus n per Box
                    try:
                        gc = ht.groupby(["__BOXKEY__","Result"]).size().rename("cnt").reset_index()
                        gb_tot = gc.groupby(["__BOXKEY__"])["cnt"].sum().rename("n").reset_index()
                        gc = gc.merge(gb_tot, on=["__BOXKEY__"], how="left")
                    except Exception:
                        gc = pd.DataFrame(columns=["__BOXKEY__","Result","cnt","n"])

                    # Index for fast lookup (tiny tables because restricted to played boxes)
                    sc_idx = {}
                    try:
                        for (_s, _b), _grp in sc.groupby(["Stream","__BOXKEY__"]):
                            _grp = _grp.sort_values(["cnt","Result"], ascending=[False, True]).reset_index(drop=True)
                            sc_idx[(str(_s), str(_b))] = _grp
                    except Exception:
                        sc_idx = {}

                    gc_idx = {}
                    try:
                        for _b, _grp in gc.groupby(["__BOXKEY__"]):
                            _grp = _grp.sort_values(["cnt","Result"], ascending=[False, True]).reset_index(drop=True)
                            gc_idx[str(_b)] = _grp
                    except Exception:
                        gc_idx = {}

                    for _stream in _streams_today:
                        _sel = []
                        try:
                            straight_candidates = []
                            for _mb in played_boxes:
                                _mb = str(_mb)
                                # Prefer stream-specific if enough samples; else fallback to global.
                                grp = sc_idx.get((str(_stream), _mb), None)
                                used = "stream"
                                n = int(grp["n"].iloc[0]) if (grp is not None and not grp.empty and "n" in grp.columns) else 0
                                if n < int(kpi_min_stream_samples):
                                    grp = gc_idx.get(_mb, None)
                                    used = "global"
                                    n = int(grp["n"].iloc[0]) if (grp is not None and not grp.empty and "n" in grp.columns) else 0
                                    if n < max(int(kpi_min_stream_samples) * 2, 50):
                                        continue

                                top_k = int(kpi_topk_per_member)
                                top = grp.head(top_k) if (grp is not None and not grp.empty) else None
                                if top is None or top.empty:
                                    continue

                                probs = []
                                for _j, _rr in top.iterrows():
                                    try:
                                        straight = str(_rr.get("Result",""))
                                        cnt = float(_rr.get("cnt", 0))
                                    except Exception:
                                        straight = ""
                                        cnt = 0.0
                                    if isinstance(straight, str) and len(straight) == 4 and straight.isdigit() and n > 0:
                                        probs.append((straight, cnt / float(n)))
                                cov = float(sum([p for (_, p) in probs]))
                                if cov < float(kpi_min_topk_cov):
                                    continue

                                for straight, p in probs:
                                    straight_candidates.append((float(p), straight, int(n), str(used), cov, _mb))

                            straight_candidates.sort(key=lambda x: (x[0], x[2], x[4]), reverse=True)
                            seen_s = set()
                            for p, straight, n, used, cov, mb in straight_candidates:
                                if straight in seen_s:
                                    continue
                                _sel.append(straight)
                                seen_s.add(straight)
                                if len(_sel) >= int(kpi_straight_cap):
                                    break
                        except Exception:
                            _sel = []
                        selected_straights_by_stream[_stream] = _sel
            for _i, _r in g.iterrows():
                stream = str(_r.get("Stream", "")).strip()
                winner = str(_r.get("Result", "")).strip()
                if not (isinstance(winner, str) and len(winner) == 4 and winner.isdigit()):
                    continue

                total_draws_eval += 1
                box_hit = 1 if box_key(winner) in set(played_box_keys) else 0
                total_box_hits += int(box_hit)
                total_box_tickets += int(boxes_per_draw)

                # Use precomputed straights for this date+stream (fast).
                selected_straights = selected_straights_by_stream.get(stream, []) if kpi_eval_straights else []
                straight_hit = 1 if (winner in set(selected_straights)) else 0
                total_straight_hits += int(straight_hit)
                total_straight_tickets += int(len(selected_straights))

                kpi_rows.append({
                    "Date": pd.to_datetime(_d).date() if hasattr(pd, "to_datetime") else _d,
                    "Stream": stream,
                    "Winner": winner,
                    "BoxesPlayed": boxes_per_draw,
                    "BoxHit": int(box_hit),
                    "StraightsPlayed": int(len(selected_straights)),
                    "StraightHit": int(straight_hit),
                })

        if not kpi_rows:
            st.warning("No draw rows were evaluated for the selected date range.")
        else:
            kpi_df = pd.DataFrame(kpi_rows)
            # Persist KPI results so they remain visible after Streamlit reruns
            try:
                st.session_state["bt_last_kpi_df"] = kpi_df.copy()
                st.session_state["bt_last_kpi_sig"] = current_sig
                st.session_state["bt_last_kpi_totals"] = {
                    "total_draws_eval": int(total_draws_eval),
                    "total_box_hits": int(total_box_hits),
                    "total_box_tickets": int(total_box_tickets),
                    "total_straight_hits": int(total_straight_hits),
                    "total_straight_tickets": int(total_straight_tickets),
                }
            except Exception:
                pass
            # Totals
            box_hit_rate = (total_box_hits / total_draws_eval) if total_draws_eval else 0.0
            straight_hit_rate = (total_straight_hits / total_straight_tickets) if total_straight_tickets else 0.0

            st.markdown("#### KPI totals")
            tot1, tot2, tot3, tot4 = st.columns(4)
            tot1.metric("Total draws evaluated", f"{total_draws_eval}")
            tot2.metric("Box hits", f"{total_box_hits} ({box_hit_rate*100:.2f}%)")
            tot3.metric("Straight tickets", f"{total_straight_tickets}")
            tot4.metric("Straight hits", f"{total_straight_hits} ({straight_hit_rate*100:.2f}%)" if total_straight_tickets else "0 (n/a)")

            # Cost modeling (simple, ticket-count based)
            try:
                _cpp = float(0.25)
            except Exception:
                _cpp = 0.25
            total_cost = _cpp * (total_box_tickets + total_straight_tickets)

            st.markdown("#### Ticket counts + cost (at $0.25 per ticket)")
            st.write({
                "TotalBoxTickets": int(total_box_tickets),
                "TotalStraightTickets": int(total_straight_tickets),
                "TotalCost($)": round(float(total_cost), 2),
            })

            st.markdown("#### Per-draw results (first 300 rows shown)")
            st.dataframe(kpi_df.head(300), use_container_width=True, hide_index=True)

            st.download_button(
                "Download KPI per-draw results (CSV)",
                data=kpi_df.to_csv(index=False).encode("utf-8"),
                file_name="backtest_box_straight_kpis.csv",
                mime="text/csv",
                key="bt_dl_box_straight_kpis",
            )


    # -------------------------
    # Strategy Finder (rows/lines)
    # -------------------------
    st.markdown("#### Strategy Finder (minimize plays)")
    st.caption(
        "Goal: find the *specific row lines* where winners concentrate most, so you can play fewer rows per core while keeping as many winners as possible."
    )

    if only_hit_days:
        st.info(
            "You have **Evaluate only days where a selected core member hit** enabled. "
            "Strategy metrics below are computed on those *hit-days only* (faster, but it can inflate day-hit rates). "
            "For true daily rates across the whole date range, re-run with that box unchecked."
        )

    # Controls
    sf1, sf2, sf3 = st.columns([1.2, 1.2, 1.2])
    with sf1:
        rank_choice = st.selectbox(
            "Which chart rows?",
            ["RankPos (overall stream position)", "BaseScoreRank (base score chart position)"],
            index=0,
            key="sf_rank_choice",
            help="RankPos is the overall stream position from the per-core stream ranking. BaseScoreRank is the rank on the BaseScore chart.",
        )
    rank_col = "RankPos" if rank_choice.startswith("RankPos") else "BaseScoreRank"

    with sf2:
        cost_per_play = st.number_input(
            "Cost per play ($)",
            min_value=0.0,
            max_value=10.0,
            value=0.25,
            step=0.05,
            key="sf_cost_per_play",
        )
    with sf3:
        member_mode = st.selectbox(
            "Member play mode (affects plays + scoring)",
            [
                "Play all 3 family members (AABC+ABBC+ABCC)",
                "Play Top2 member picks (requires tracking)",
                "Play Top1 member pick (requires tracking)",
            ],
            index=0,
            key="sf_member_mode",
            help="All-3 counts a win whenever the core hit in that stream and you played the stream. Top2/Top1 count wins only if the predicted member(s) match the actual member.",
        )

    # Determine member multiplier + scoring filter
    member_mult = 3
    member_filter_col = None
    if member_mode.startswith("Play Top2"):
        member_mult = 2
        member_filter_col = "MemberHitTop2"
    elif member_mode.startswith("Play Top1"):
        member_mult = 1
        member_filter_col = "MemberHitTop1"

    # If member mode selected but tracking not available, fall back safely
    if member_filter_col is not None:
        if (not member_track) or (member_filter_col not in out.columns):
            st.warning("Top1/Top2 scoring requires **Track member accuracy**. Falling back to **Play all 3** for Strategy Finder.")
            member_mult = 3
            member_filter_col = None

    # Prepare rank dataframe
    df_rank = out.copy()
    if rank_col not in df_rank.columns:
        st.warning(f"Strategy Finder needs column '{rank_col}', but it was not found in backtest output.")
        df_rank = pd.DataFrame()
    else:
        df_rank[rank_col] = pd.to_numeric(df_rank[rank_col], errors="coerce").astype("Int64")
        df_rank = df_rank.dropna(subset=[rank_col])
        df_rank[rank_col] = df_rank[rank_col].astype(int)

    if df_rank.empty:
        st.info("No ranked rows available to analyze for Strategy Finder.")
    else:
        # Apply member scoring filter if requested
        if member_filter_col is not None and member_filter_col in df_rank.columns:
            df_rank[member_filter_col] = df_rank[member_filter_col].fillna(False).astype(bool)
            df_rank_scored = df_rank[df_rank[member_filter_col]].copy()
        else:
            df_rank_scored = df_rank

        total_wins = len(df_rank_scored)
        total_days = int(df_rank_scored["Date"].nunique())
        cores_in_test = sorted(df_rank_scored["Core"].astype(str).unique().tolist())
        ncores = len(cores_in_test)

        if total_wins == 0:
            st.info("No wins are scorable under the selected member mode in this window.")
        else:
            # Row hotness table
            rc = df_rank_scored.groupby(rank_col).agg(
                Wins=("Core", "size"),
                DaysWithWin=("Date", "nunique"),
            ).reset_index().rename(columns={rank_col: "Row"})
            rc["WinPct"] = (rc["Wins"] / total_wins * 100).round(2)
            rc["DayHitPct"] = (rc["DaysWithWin"] / max(1, total_days) * 100).round(2)

            rc = rc.sort_values(["Wins", "DaysWithWin", "Row"], ascending=[False, False, True])

            hottest_row = int(rc.iloc[0]["Row"])
            hottest_wins = int(rc.iloc[0]["Wins"])
            hottest_dayhit = int(rc.iloc[0]["DaysWithWin"])
            # Plays/day assumes you play this row for every tested core every day
            plays_per_day_row1 = ncores * 1 * member_mult
            cost_per_day_row1 = plays_per_day_row1 * float(cost_per_play)
            st.markdown(
                f"**Hottest single row:** Row **{hottest_row}** on **{rank_col}** "
                f"captured **{hottest_wins}/{total_wins} wins** ({(hottest_wins/total_wins*100):.1f}%), "
                f"and hit on **{hottest_dayhit}/{total_days} days** ({(hottest_dayhit/max(1,total_days)*100):.1f}%). "
                f"Playing only that row across **{ncores} cores** costs ~**{plays_per_day_row1} plays/day** (≈ ${cost_per_day_row1:,.2f}/day at ${cost_per_play:.2f})."
            )

            with st.expander("Row hotness table (all rows)", expanded=False):
                st.dataframe(rc, use_container_width=True, hide_index=True)

            # Evaluate top-K row strategies (specific line sets, not ranges)
            max_row = int(rc["Row"].max())
            max_k_default = min(9, max(1, min(15, max_row)))
            k_max = st.slider(
                "Evaluate Top‑K hottest rows (specific row lines)",
                min_value=1,
                max_value=min(15, max_row),
                value=max_k_default,
                step=1,
                key="sf_kmax",
                help="Top‑K is built from the K hottest rows by win count (not a contiguous range).",
            )

            top_rows = rc["Row"].astype(int).tolist()

            strat_rows = []
            for k in range(1, int(k_max) + 1):
                rows_k = top_rows[:k]
                sub = df_rank_scored[df_rank_scored["Core"].astype(str).isin(cores_in_test) & df_rank_scored[rank_col].isin(rows_k)]
                cap_wins = int(len(sub))
                cap_days = int(sub["Date"].nunique())
                cap_pct = (cap_wins / total_wins * 100.0) if total_wins else 0.0
                day_pct = (cap_days / max(1, total_days) * 100.0)

                plays_per_day = ncores * k * member_mult
                cost_per_day = plays_per_day * float(cost_per_play)
                # Over the tested days, how much spend per captured win?
                spend_total = cost_per_day * total_days
                cost_per_win = (spend_total / cap_wins) if cap_wins > 0 else None
                strat_rows.append({
                    "K (rows)": k,
                    "Rows (specific lines)": ",".join(str(r) for r in rows_k),
                    "CapturedWins": cap_wins,
                    "CapturePct": round(cap_pct, 2),
                    "DaysWith≥1Win": cap_days,
                    "DayHitPct": round(day_pct, 2),
                    "Plays/Day": int(plays_per_day),
                    "Cost/Day($)": round(cost_per_day, 2),
                    "Cost/CapturedWin($)": (round(cost_per_win, 2) if cost_per_win is not None else None),
                })

            strat_df = pd.DataFrame(strat_rows)
            st.markdown("##### Top‑K row strategies (play these specific lines for every tested core)")
            st.dataframe(strat_df, use_container_width=True, hide_index=True)

            # Manual selection (exact rows)
            st.markdown("##### Try a custom set of row lines")
            default_manual = top_rows[:min(3, len(top_rows))]
            manual_rows = st.multiselect(
                "Select specific rows to play (exact lines, not ranges)",
                options=sorted(top_rows),
                default=default_manual,
                key="sf_manual_rows",
            )
            if manual_rows:
                subm = df_rank_scored[df_rank_scored[rank_col].isin([int(x) for x in manual_rows])].copy()
                cap_wins_m = int(len(subm))
                cap_days_m = int(subm["Date"].nunique())
                plays_per_day_m = ncores * len(manual_rows) * member_mult
                cost_per_day_m = plays_per_day_m * float(cost_per_play)
                st.write(
                    f"Custom rows captured **{cap_wins_m}/{total_wins} wins** ({(cap_wins_m/total_wins*100):.1f}%) "
                    f"across **{cap_days_m}/{total_days} days** ({(cap_days_m/max(1,total_days)*100):.1f}%). "
                    f"Plays/day = **{plays_per_day_m}** (≈ ${cost_per_day_m:,.2f}/day)."
                )

                # Per-core breakdown for the chosen rows
                pc = subm.groupby("Core").size().reset_index(name="CapturedWins")
                total_by_core = df_rank_scored.groupby("Core").size().reset_index(name="TotalWins")
                pc = pc.merge(total_by_core, on="Core", how="right").fillna({"CapturedWins": 0})
                pc["CapturePct"] = (pc["CapturedWins"] / pc["TotalWins"] * 100).round(1)
                pc = pc.sort_values(["CapturePct", "TotalWins"], ascending=[False, False])
                with st.expander("Per-core capture for these rows", expanded=False):
                    st.dataframe(pc, use_container_width=True, hide_index=True)
            else:
                st.info("Select at least one row to see custom strategy metrics.")

            # Core-by-core Top2 member recommendation quick table (for the current backtest window)
            if member_track and ("ActualFamilyMember" in out.columns) and (not out.dropna(subset=["ActualFamilyMember"]).empty):
                st.markdown("##### Core-by-core Top2 member recommendation (from training window)")
                st.caption("This summarizes which member label (AABC/ABBC/ABCC) actually hit most often in this backtest window.")
                md = out.dropna(subset=["ActualFamilyMember"]).copy()
                dist = md.pivot_table(index="Core", columns="ActualFamilyMember", values="Date", aggfunc="size", fill_value=0)
                for col in ["AABC","ABBC","ABCC"]:
                    if col not in dist.columns:
                        dist[col] = 0
                dist = dist[["AABC","ABBC","ABCC"]]
                dist["Total"] = dist.sum(axis=1)
                # Top2 members
                def _top2(row):
                    pairs = [(k, int(row[k])) for k in ["AABC","ABBC","ABCC"]]
                    pairs.sort(key=lambda x: (-x[1], x[0]))
                    return pairs[0][0], pairs[1][0]
                top2 = dist.apply(_top2, axis=1, result_type="expand")
                dist["Top1"] = top2[0]
                dist["Top2"] = top2[1]
                dist["Top1Pct"] = dist.apply(lambda r: round((int(r[r["Top1"]]) / (int(r["Total"]) or 1)) * 100, 1), axis=1)
                dist = dist.reset_index().sort_values(["Top1Pct","Total"], ascending=[False, False])
                st.dataframe(dist[["Core","AABC","ABBC","ABCC","Total","Top1","Top2","Top1Pct"]], use_container_width=True, hide_index=True)



# -------------------------
# Member strategy comparisons (walk-forward, no cheat)
# -------------------------
if member_track and (not out.empty):
    st.markdown("#### Member Strategy Finder (MODE vs LAST vs overrides)")
    st.caption(
        "These comparisons are **walk-forward safe**: for each test_date, member predictions are generated using only rows with Date < test_date. "
        "This helps decide whether you should play 1 member, 2 members, or all 3 for a given core."
    )

    # Compute only on rows where the winner was one of the family members (AABC/ABBC/ABCC)
    mc = out.dropna(subset=["ActualFamilyMember"]).copy()
    mc = mc[mc["ActualFamilyMember"].astype(str).isin(["AABC","ABBC","ABCC"])].copy()

    if mc.empty:
        st.info("No family-member rows in this backtest window to compare member strategies.")
    else:
        # Build predictions per row under multiple strategies
        preds = []
        for i, r in mc.iterrows():
            try:
                core = str(_safe_get(r,"Core","")).zfill(3)
                stream = str(_safe_get(r,"Stream",""))
                td = pd.to_datetime(_safe_get(r,"Date"))
                variants = _member_prediction_variants(
                    df_all=df_all,
                    traits_pos_df=seed_traits_pos_df,
                    core_key=core,
                    test_date=td,
                    window_days=int(cfg.window_days),
                    stream=stream,
                    basis=str(member_basis),
                )
                preds.append(variants)
            except Exception:
                preds.append({"MODE": None, "LAST_GLOBAL": None, "LAST_HIER": None, "SEED_OVERRIDE": None, "TRAIT_OVERRIDE": None})

        var_df = pd.DataFrame(preds)
        for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]:
            mc[f"PredMember_{c}"] = var_df[c].values
            mc[f"Hit_{c}"] = (mc["ActualFamilyMember"].astype(str) == mc[f"PredMember_{c}"].astype(str))

        # Summary by core
        sum_rows = []
        for core, g in mc.groupby("Core"):
            n = int(len(g))
            row = {"Core": core, "N": n}
            for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]:
                row[f"Top1_{c}"] = int(g[f"Hit_{c}"].sum())
                row[f"Top1Rate_{c}"] = round(float(g[f"Hit_{c}"].mean()), 4) if n else 0.0
            sum_rows.append(row)
        sum_df = pd.DataFrame(sum_rows).sort_values(["N"], ascending=False)

        st.markdown("##### Top1 member accuracy by core (compare strategies)")
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        # Overall summary
        overall = {"Metric": ["Rows (family-member only)"]}
        overall["Value"] = [len(mc)]
        overall_df = pd.DataFrame(overall)
        st.dataframe(overall_df, use_container_width=True, hide_index=True)

        # Recommended Top2 members per core (latest as-of end_dt)
        st.markdown("##### Core-by-core Top2 member recommendations (for play reduction)")
        st.caption(
            "Top2 is built from the **training window right before the most recent test_date** in this run. "
            "Use this when Top1 is weak but Top2 is strong (play 2 members instead of all 3)."
        )

        try:
            last_test = pd.to_datetime(out["Date"], errors="coerce").max()
        except Exception:
            last_test = pd.to_datetime(end_dt)

        recs = []
        for core in sorted(set(mc["Core"].astype(str).tolist())):
            # MODE distribution from the existing predictor (returns top1/top2 by counts)
            mp = predict_core_member(df_all, core, last_test, int(cfg.window_days), basis=("core_stream" if str(member_basis)=="core_stream" else "core"), stream=None, include_rare=False)
            t1, t2 = mp.get("top1"), mp.get("top2")
            ntrain = int(mp.get("n") or 0)
            recs.append({"Core": core, "Top1(MODE)": t1, "Top2(MODE)": t2, "TrainN": ntrain})
        rec_df = pd.DataFrame(recs).sort_values(["TrainN"], ascending=False)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        with st.expander("Download member strategy comparison rows (copy/paste ready)", expanded=False):
            dl_cols = ["Date","Stream","Core","Winner","ActualFamilyMember"] +                           [f"PredMember_{c}" for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]] +                           [f"Hit_{c}" for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]]
            dl_cols = [c for c in dl_cols if c in mc.columns]
            st.dataframe(mc[dl_cols].sort_values(["Date","Stream","Core"]), use_container_width=True)
            st.download_button(
                "Download member strategy rows CSV",
                data=mc[dl_cols].to_csv(index=False).encode("utf-8"),
                file_name="member_strategy_comparisons.csv",
                mime="text/csv",
            )

    st.markdown("#### Hit/Miss detail (copy/paste ready)")
    show_cols = [
        "Date", "Stream", "Core", "Winner", "Predicted", "Bucket", "RankPos", "BaseScoreRank", "HitsWindow", "DaysSinceLastHit",
        "ActualFamilyMember", "PredMemberTop1", "PredMemberTop2", "MemberHitTop1", "MemberHitTop2", "MemberTrainN", "AsOfMaxDate",
    ]
    safe_cols = [c for c in show_cols if c in out.columns]
    st.dataframe(out[safe_cols].sort_values(["Date","Stream","Core"], ascending=True), use_container_width=True)
    st.download_button(
        "Download walk-forward rows CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="walkforward_rows.csv",
        mime="text/csv",
    )

    st.markdown("#### By core")
    by_core = out.groupby("Core").agg(Total=("Predicted","size"), Predicted=("Predicted","sum")).reset_index()
    by_core["PredictedPct"] = (by_core["Predicted"] / by_core["Total"] * 100).round(1)
    st.dataframe(by_core.sort_values(["PredictedPct","Total"], ascending=[False, False]), use_container_width=True, hide_index=True)

    st.markdown("#### By stream")
    by_stream = out.groupby("Stream").agg(Total=("Predicted","size"), Predicted=("Predicted","sum")).reset_index()
    by_stream["PredictedPct"] = (by_stream["Predicted"] / by_stream["Total"] * 100).round(1)
    st.dataframe(by_stream.sort_values(["Predicted","Total"], ascending=[False, False]).head(80), use_container_width=True, hide_index=True)

    # Manual day replay
    st.markdown("#### Manual day replay (mock what you'd do daily)")
    unique_days = sorted(out["Date"].unique())
    sel_day = st.selectbox("Pick a day to inspect", options=unique_days, index=max(0, len(unique_days)-1), key="bt_replay_day")
    sel_day_ts = pd.Timestamp(sel_day)

    # Show that day's core hits and where they sat on the chart
    day_rows = out[out["Date"] == sel_day].copy()
    st.write(f"Core-family wins on {sel_day}: {len(day_rows)}")
    st.dataframe(day_rows.sort_values(["Core","Predicted"], ascending=[True, False]), use_container_width=True, hide_index=True)

    # For each core that hit: show predicted stream buckets for that day (from training)
    for core in sorted(day_rows["Core"].unique()):
        st.markdown(f"**Core {core}: predicted streams as of {sel_day}**")

        if member_track:
            try:
                mp_overall = predict_core_member(
                    df_all,
                    core,
                    pd.to_datetime(sel_day_ts).normalize(),
                    window_days=int(cfg.window_days),
                    basis="core",
                    include_rare=False,
                )
            except Exception:
                mp_overall = {}
            if mp_overall:
                st.caption(
                    f"Member pick (overall): Top1={mp_overall.get('top1')}, Top2={mp_overall.get('top2')} (train hits={mp_overall.get('n')})"
                )
        train_df = df_all[df_all["Date"] < sel_day_ts]
        stats_df = compute_stream_stats(train_df, core, window_days=int(cfg.window_days))
        buckets = bucket_recommendations(stats_df, cfg)
        base_df = buckets.get("Top12BaseScore", pd.DataFrame()).copy()
        due_df = buckets.get("Due8", pd.DataFrame()).copy()
        if not base_df.empty:
            base_df["Bucket"] = "BaseScore"
        if not due_df.empty:
            due_df["Bucket"] = "Due8"
        pred_df = pd.concat([base_df, due_df], ignore_index=True)
        if pred_df.empty:
            st.info("No bucket recommendations for this core/day.")
            continue
        # mark if this stream was an actual win that day
        win_streams = set(day_rows[day_rows["Core"] == core]["Stream"].astype(str).tolist())
        pred_df["WonThatDay"] = pred_df["Stream"].astype(str).isin(win_streams)
        cols = [c for c in ["Bucket","Stream","RankPos","BaseScoreRank","HitsWindow","DaysSinceLastHit","WonThatDay"] if c in pred_df.columns]
        st.dataframe(pred_df[cols].sort_values(["WonThatDay","Bucket"], ascending=[False, True]), use_container_width=True, hide_index=True)



# ------------------------------------------------------------
# Validator: Walk-forward Top2 member playlist (AABC only)
# ------------------------------------------------------------

def _core_key_from_box_aabc(box4: str) -> str | None:
    """For an AABC family-doubles box key (sorted 4 digits), return the 3-digit core key (sorted unique digits)."""
    try:
        s = str(box4).strip()
        if len(s) != 4 or not s.isdigit():
            return None
        uniq = sorted(set(s))
        if len(uniq) != 3:
            return None
        return "".join(uniq)
    except Exception:
        return None

@st.cache_data(show_spinner=False, max_entries=32)
def _validator_prepare_family_df(df_all: pd.DataFrame) -> pd.DataFrame:
    """Prepare a compact dataframe for AABC-only validation (Top2 members).
    No simulations: uses real df_all rows only.
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=["Date","State","Game","Stream","BoxKey4","CoreKey","MemberLabel"])
    d = df_all.copy()
    # Ensure required columns exist
    for col in ["Date","State","Game","Stream","BoxKey4","Structure","Result"]:
        if col not in d.columns:
            # Attempt to derive minimal compatibility fields
            if col == "Stream" and ("State" in d.columns and "Game" in d.columns):
                d["Stream"] = d["State"].astype(str).str.strip() + " | " + d["Game"].astype(str).str.strip()
            elif col == "BoxKey4" and "Result" in d.columns:
                d["BoxKey4"] = d["Result"].astype(str).str.zfill(4).map(box_key)
            elif col == "Structure" and "Result" in d.columns:
                d["Structure"] = d["Result"].astype(str).str.zfill(4).map(structure_of_4)
            elif col == "Result" and "Pick4" in d.columns:
                d["Result"] = d["Pick4"]
            else:
                # If still missing, bail with empty
                return pd.DataFrame(columns=["Date","State","Game","Stream","BoxKey4","CoreKey","MemberLabel"])
    # Normalize date
    d["Date"] = _parse_date_series(d["Date"])
    d = d[d["Date"].notna()].copy()
    # AABC only (family doubles)
    d = d[d["Structure"].astype(str).eq("AABC")].copy()
    if d.empty:
        return pd.DataFrame(columns=["Date","State","Game","Stream","BoxKey4","CoreKey","MemberLabel"])
    d["BoxKey4"] = d["BoxKey4"].astype(str)
    d["CoreKey"] = d["BoxKey4"].map(_core_key_from_box_aabc)
    d = d[d["CoreKey"].notna()].copy()
    # Member label (AABC/ABBC/ABCC) by core map
    d["MemberLabel"] = [
        core_member_label(ck, bk, include_rare=False) for ck, bk in zip(d["CoreKey"].tolist(), d["BoxKey4"].tolist())
    ]
    return d[["Date","State","Game","Stream","BoxKey4","CoreKey","MemberLabel"]].sort_values("Date").reset_index(drop=True)

def _validator_top2_members_map(df_train: pd.DataFrame) -> pd.DataFrame:
    """Learn Top2 members per (CoreKey, Stream) from training window."""
    if df_train is None or df_train.empty:
        return pd.DataFrame(columns=["CoreKey","Stream","Top1Box","Top2Box","Top1Cnt","Top2Cnt","TrainN","TrainCnt_AABC","TrainCnt_ABBC","TrainCnt_ABCC"])
    g = df_train.groupby(["CoreKey","Stream","BoxKey4"], as_index=False).size()
    g = g.rename(columns={"size":"Cnt"})
    # Per (CoreKey, Stream) choose top2 by count then by BoxKey4 for stability
    g = g.sort_values(["CoreKey","Stream","Cnt","BoxKey4"], ascending=[True, True, False, True])
    top2 = g.groupby(["CoreKey","Stream"]).head(2).copy()
    # Pivot into Top1/Top2 columns
    def _pack(grp: pd.DataFrame) -> dict:
        rows = grp.sort_values(["Cnt","BoxKey4"], ascending=[False, True]).to_dict("records")
        top1 = rows[0] if len(rows) > 0 else None
        top2r = rows[1] if len(rows) > 1 else None
        return {
            "Top1Box": (top1["BoxKey4"] if top1 else None),
            "Top2Box": (top2r["BoxKey4"] if top2r else None),
            "Top1Cnt": int(top1["Cnt"]) if top1 else 0,
            "Top2Cnt": int(top2r["Cnt"]) if top2r else 0,
        }
    packed = top2.groupby(["CoreKey","Stream"]).apply(_pack).apply(pd.Series).reset_index()
    # Training label counts (diagnostics)
    lab_counts = df_train.pivot_table(index=["CoreKey","Stream"], columns="MemberLabel", values="BoxKey4", aggfunc="size", fill_value=0).reset_index()
    # Ensure expected cols
    for c in ["AABC","ABBC","ABCC"]:
        if c not in lab_counts.columns:
            lab_counts[c] = 0
    trainN = df_train.groupby(["CoreKey","Stream"], as_index=False).size().rename(columns={"size":"TrainN"})
    out = packed.merge(trainN, on=["CoreKey","Stream"], how="left").merge(lab_counts[["CoreKey","Stream","AABC","ABBC","ABCC"]], on=["CoreKey","Stream"], how="left")
    out = out.rename(columns={"AABC":"TrainCnt_AABC","ABBC":"TrainCnt_ABBC","ABCC":"TrainCnt_ABCC"})
    return out

def _validator_hits_by_pair(df_window: pd.DataFrame, top2_map: pd.DataFrame) -> pd.DataFrame:
    """Compute captured hits in a window per (CoreKey, Stream) using learned Top2 boxes."""
    if df_window is None or df_window.empty or top2_map is None or top2_map.empty:
        return pd.DataFrame(columns=["CoreKey","Stream","Hits"])
    m = top2_map[["CoreKey","Stream","Top1Box","Top2Box"]].copy()
    # Build a set membership test by expanding to long form
    long = pd.concat([
        m.rename(columns={"Top1Box":"BoxKey4"})[["CoreKey","Stream","BoxKey4"]],
        m.rename(columns={"Top2Box":"BoxKey4"})[["CoreKey","Stream","BoxKey4"]],
    ], ignore_index=True)
    long = long[long["BoxKey4"].notna()].copy()
    long["BoxKey4"] = long["BoxKey4"].astype(str)
    w = df_window.copy()
    w["BoxKey4"] = w["BoxKey4"].astype(str)
    merged = w.merge(long, on=["CoreKey","Stream","BoxKey4"], how="inner")
    hits = merged.groupby(["CoreKey","Stream"], as_index=False).size().rename(columns={"size":"Hits"})
    return hits

def _validator_build_min_playlist(hits_slices: list[pd.DataFrame], target_hits: int) -> list[tuple[str,str]]:
    """Greedy minimal playlist: add (CoreKey,Stream) pairs until each slice reaches target_hits."""
    # Build candidate universe
    all_pairs = set()
    for hs in hits_slices:
        if hs is None or hs.empty:
            continue
        for ck, st_ in zip(hs["CoreKey"].tolist(), hs["Stream"].tolist()):
            all_pairs.add((str(ck), str(st_)))
    all_pairs = sorted(all_pairs)
    if not all_pairs:
        return []
    # Convert each slice hits to dict
    slice_dicts = []
    for hs in hits_slices:
        d = {}
        if hs is not None and not hs.empty:
            d = {(str(ck), str(st_)): int(h) for ck, st_, h in zip(hs["CoreKey"], hs["Stream"], hs["Hits"])}
        slice_dicts.append(d)
    chosen: set[tuple[str,str]] = set()
    cur = [0]*len(slice_dicts)

    # Greedy loop
    # Score: maximize improvement to the minimum-satisfied slice (lexicographically)
    while True:
        if all(v >= target_hits for v in cur):
            break
        best = None
        best_score = None
        best_new = None
        for pair in all_pairs:
            if pair in chosen:
                continue
            new = []
            for i,d in enumerate(slice_dicts):
                new.append(cur[i] + int(d.get(pair, 0)))
            # Primary: improve the minimum slice hits
            score = (min(new), sum(new))
            if best_score is None or score > best_score:
                best_score = score
                best = pair
                best_new = new
        if best is None:
            break
        chosen.add(best)
        cur = best_new
        # Stop if no progress possible (avoid infinite loops)
        if best_score is not None and best_score[0] == min(cur) and best_score[1] == sum(cur) and all(slice_dicts[i].get(best,0)==0 for i in range(len(slice_dicts))):
            break
        # Hard cap to avoid runaway (still allows a lot; protects UI)
        if len(chosen) > 5000:
            break
    return sorted(chosen)

@st.cache_data(show_spinner=False, max_entries=8)
def run_validator_walk_forward(
    df_family: pd.DataFrame,
    train_days: int = 180,
    refresh_days: int = 7,
    test_days: int = 60,
    target_hits: int = 60,
    cores_filter: tuple[str, ...] = (),
    streams_filter: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Run 3-year walk-forward validator using Top2 member learning.
    Returns summary per refresh anchor.
    """
    if df_family is None or df_family.empty:
        return pd.DataFrame()
    d = df_family.copy()
    if cores_filter:
        cores_set = set(str(c).zfill(3) for c in cores_filter)
        d = d[d["CoreKey"].isin(cores_set)].copy()
    if streams_filter:
        streams_set = set(streams_filter)
        d = d[d["Stream"].isin(streams_set)].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values("Date").reset_index(drop=True)
    min_date = d["Date"].min().normalize()
    max_date = d["Date"].max().normalize()

    # First anchor must have train_days history and then test_days ahead
    start_anchor = min_date + pd.Timedelta(days=train_days)
    end_anchor = max_date - pd.Timedelta(days=test_days)
    if start_anchor > end_anchor:
        return pd.DataFrame()

    anchors = pd.date_range(start=start_anchor, end=end_anchor, freq=f"{int(refresh_days)}D")
    out_rows = []

    for anchor in anchors:
        train_start = anchor - pd.Timedelta(days=train_days)
        train_end = anchor - pd.Timedelta(days=1)
        test_start = anchor
        test_end = anchor + pd.Timedelta(days=test_days) - pd.Timedelta(days=1)

        df_train = d[(d["Date"] >= train_start) & (d["Date"] <= train_end)]
        if df_train.empty:
            continue

        top2_map = _validator_top2_members_map(df_train)

        # Training slices (3 contiguous 60d blocks ending at train_end)
        s3_end = train_end
        s3_start = s3_end - pd.Timedelta(days=test_days) + pd.Timedelta(days=1)
        s2_end = s3_start - pd.Timedelta(days=1)
        s2_start = s2_end - pd.Timedelta(days=test_days) + pd.Timedelta(days=1)
        s1_end = s2_start - pd.Timedelta(days=1)
        s1_start = s1_end - pd.Timedelta(days=test_days) + pd.Timedelta(days=1)

        df_s1 = d[(d["Date"] >= s1_start) & (d["Date"] <= s1_end)]
        df_s2 = d[(d["Date"] >= s2_start) & (d["Date"] <= s2_end)]
        df_s3 = d[(d["Date"] >= s3_start) & (d["Date"] <= s3_end)]
        df_test = d[(d["Date"] >= test_start) & (d["Date"] <= test_end)]

        hits_s1 = _validator_hits_by_pair(df_s1, top2_map)
        hits_s2 = _validator_hits_by_pair(df_s2, top2_map)
        hits_s3 = _validator_hits_by_pair(df_s3, top2_map)

        playlist = _validator_build_min_playlist([hits_s1, hits_s2, hits_s3], target_hits=target_hits)
        # Compute realized hits in test window for this playlist
        if playlist:
            pl_df = pd.DataFrame(playlist, columns=["CoreKey","Stream"])
            # Expand Top2 to long for playlist only
            top2_pl = top2_map.merge(pl_df, on=["CoreKey","Stream"], how="inner")
            hits_test_pairs = _validator_hits_by_pair(df_test, top2_pl)
            test_hits = int(hits_test_pairs["Hits"].sum()) if not hits_test_pairs.empty else 0
            # Activations: number of draw rows in test for chosen streams/cores (opportunities)
            opp = int(df_test.merge(pl_df, on=["CoreKey","Stream"], how="inner").shape[0])
            plays = opp * 2  # Top2 members only
        else:
            test_hits = 0
            opp = 0
            plays = 0

        # Training slice totals for reporting (using playlist)
        if playlist:
            pl_df = pd.DataFrame(playlist, columns=["CoreKey","Stream"])
            top2_pl = top2_map.merge(pl_df, on=["CoreKey","Stream"], how="inner")
            tr1 = int(_validator_hits_by_pair(df_s1, top2_pl)["Hits"].sum()) if not df_s1.empty else 0
            tr2 = int(_validator_hits_by_pair(df_s2, top2_pl)["Hits"].sum()) if not df_s2.empty else 0
            tr3 = int(_validator_hits_by_pair(df_s3, top2_pl)["Hits"].sum()) if not df_s3.empty else 0
        else:
            tr1 = tr2 = tr3 = 0

        out_rows.append({
            "AnchorDate": anchor.date().isoformat(),
            "TrainStart": train_start.date().isoformat(),
            "TrainEnd": train_end.date().isoformat(),
            "TestStart": test_start.date().isoformat(),
            "TestEnd": test_end.date().isoformat(),
            "PlaylistPairs": int(len(playlist)),
            "TestHitsCaptured": int(test_hits),
            "Pass_>=Target": bool(test_hits >= target_hits),
            "TestOpportunities": int(opp),
            "TestPlays_Top2": int(plays),
            "TrainSlice1Hits": int(tr1),
            "TrainSlice2Hits": int(tr2),
            "TrainSlice3Hits": int(tr3),
        })

    return pd.DataFrame(out_rows)

def render_validator_tab(df_all: pd.DataFrame) -> None:
    st.subheader("Validator — Walk-forward Top2 (AABC only)")
    st.caption(
        "Honest validation for the minimize-plays strategy: "
        "a 'hit' counts only when the winning draw is AABC and matches one of the Top2 family members learned per (core, stream) from the training window."
    )

    if df_all is None or df_all.empty:
        st.warning("Upload/load a history file first.")
        return

    df_family = _validator_prepare_family_df(df_all)
    if df_family.empty:
        st.warning("No AABC (family doubles) rows found after parsing. Validator needs AABC rows to evaluate Top2.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        train_days = st.number_input("Train days", min_value=60, max_value=365, value=180, step=1)
    with col2:
        test_days = st.number_input("Test days", min_value=30, max_value=120, value=60, step=1)
    with col3:
        refresh_days = st.number_input("Refresh cadence (days)", min_value=1, max_value=60, value=7, step=1)
    with col4:
        target_hits = st.number_input("Target hits per test window", min_value=1, max_value=500, value=60, step=1)

    with st.expander("Filters (optional)", expanded=False):
        cores = sorted(df_family["CoreKey"].unique().tolist())
        streams = sorted(df_family["Stream"].unique().tolist())
        sel_cores = st.multiselect("Limit to cores (optional)", options=cores, default=[], help="Leave empty for ALL cores in history.")
        sel_streams = st.multiselect("Limit to streams (optional)", options=streams, default=[], help="Leave empty for ALL streams in history.")

    run_btn = st.button("Run 3-year walk-forward validator", type="primary", key="run_validator_btn")
    if not run_btn:
        st.info("Set your parameters, then click **Run** to compute pass rate and playlist size over time.")
        return

    with st.spinner("Running walk-forward validator... (cached; first run may take a bit)"):
        res = run_validator_walk_forward(
            df_family=df_family,
            train_days=int(train_days),
            refresh_days=int(refresh_days),
            test_days=int(test_days),
            target_hits=int(target_hits),
            cores_filter=tuple(sel_cores),
            streams_filter=tuple(sel_streams),
        )

    if res is None or res.empty:
        st.warning("Validator produced no results (not enough data for the requested windows, or filters too restrictive).")
        return

    pass_rate = float(res["Pass_>=Target"].mean()) if "Pass_>=Target" in res.columns else 0.0
    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Pass rate", f"{pass_rate*100:.1f}%")
    colm2.metric("Median test hits", f"{int(res['TestHitsCaptured'].median())}")
    colm3.metric("Median playlist pairs", f"{int(res['PlaylistPairs'].median())}")
    colm4.metric("Worst test hits", f"{int(res['TestHitsCaptured'].min())}")

    st.markdown("#### Walk-forward results (per refresh anchor)")
    st.dataframe(res, use_container_width=True)

    # Copy/paste summary
    st.markdown("#### Copy/paste summary")
    summary = (
        f"train={int(train_days)}d | test={int(test_days)}d | refresh={int(refresh_days)}d | target={int(target_hits)}\n"
        f"pass_rate={pass_rate*100:.1f}% | median_hits={int(res['TestHitsCaptured'].median())} | median_pairs={int(res['PlaylistPairs'].median())}\n"
        f"worst_hits={int(res['TestHitsCaptured'].min())} | best_hits={int(res['TestHitsCaptured'].max())}"
    )
    st.code(summary, language="text")



def _age_days(ts: Optional[pd.Timestamp]) -> Optional[int]:
    if ts is None or pd.isna(ts):
        return None
    try:
        d = pd.to_datetime(ts).date()
        return (today - d).days
    except Exception:
        return None

st.sidebar.markdown("### Data freshness")
if last_all is not None and not pd.isna(last_all):
    st.sidebar.caption(f"All‑states history most recent date: {pd.to_datetime(last_all).date()}  (age: {_age_days(last_all)} days)")
else:
    st.sidebar.caption("All‑states history most recent date: (not found)")

if last_24 is not None and not pd.isna(last_24):
    st.sidebar.caption(f"24h map most recent date: {pd.to_datetime(last_24).date()}  (age: {_age_days(last_24)} days)")
else:
    st.sidebar.caption("24h map most recent date: (not uploaded)")

st.sidebar.caption(f"Tip: if ages are >1–2 days, your files are probably behind.")

if master_file is None:
    st.info("Upload your all‑states history file to start.")
    df_all = pd.DataFrame()  # allow the app to render tabs/sections before upload

if master_file is not None and df_all.empty:
    st.error("Could not parse your history file. Make sure it contains Date, State, Game, Results.")
    st.stop()

# One place to show dataset info
colA, colB, colC = st.columns(3)

# Guard against the pre-upload state (empty df_all with no columns)
rows_n = int(len(df_all)) if df_all is not None else 0
streams_n = int(df_all["Stream"].nunique()) if (df_all is not None and ("Stream" in df_all.columns)) else 0

with colA:
    st.metric("Rows (draws)", f"{rows_n:,}")
with colB:
    st.metric("Streams", f"{streams_n:,}")
with colC:
    if df_all is not None and ("Date" in df_all.columns) and (not df_all.empty):
        try:
            min_d = df_all["Date"].min().date().isoformat()
            max_d = df_all["Date"].max().date().isoformat()
            st.caption(f"Date span: {min_d} → {max_d}")
        except Exception:
            st.caption("Date span: —")
    else:
        st.caption("Date span: —")



st.divider()


# Load latest walk-forward backtest output (if any) for planner panels
out = st.session_state.get("wf_backtest_out", pd.DataFrame())

# -------------------------
# Auto Profit Planner (Box baseline + selective Straight booster)
# -------------------------
if (not out.empty):
    st.markdown("#### Auto Profit Planner (make the strategy automatic)")
    st.caption(
        "This planner uses your **walk-forward backtest rows** (no leakage) to recommend a small set of cores "
        "that can reach your target **average box wins per day**, while minimizing plays. "
        "Straight/Box combo tickets are supported as an optional **selective straight booster**."
    )

    with st.expander("Open Auto Profit Planner", expanded=False):
        # --- Controls ---
        ap1, ap2, ap3, ap4 = st.columns([1.15, 1.0, 1.0, 1.0])
        with ap1:
            eval_days = st.number_input("Evaluate on last N days", min_value=30, max_value=365, value=90, step=10, key="ap_eval_days")
        with ap2:
            target_avg_box_wins = st.number_input("Target avg BOX wins/day", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="ap_target_box_wins")
        with ap3:
            max_box_plays_per_day = st.number_input("Max BOX plays/day (soft cap)", min_value=1, max_value=2000, value=120, step=5, key="ap_max_box_plays_day")
        with ap4:
            cost_per_play_ap = st.number_input("Cost per play ($)", min_value=0.0, max_value=10.0, value=float(cost_per_play), step=0.05, key="ap_cost_per_play")

        ap5, ap6, ap7 = st.columns([1.2, 1.2, 1.2])
        with ap5:
            rank_basis = st.selectbox(
                "Row basis (same meaning as Strategy Finder)",
                ["RankPos", "BaseScoreRank"],
                index=0,
                key="ap_rank_basis",
                help="This controls which row-number column is used to define your 'line' selections.",
            )
        rank_col_ap = "RankPos" if rank_basis == "RankPos" else "BaseScoreRank"

        with ap6:
            # Rows/lines selection
            default_rows = list(range(1, 13))  # Top12 baseline
            auto_rows = st.checkbox(
                f"Auto-expand {rank_col_ap} rows to meet target (recommended)",
                value=True,
                key="ap_auto_rows",
                help="If enabled, the planner will choose the smallest set of rows/lines (in priority order) needed to reach your target avg BOX wins/day.",
            )
            if auto_rows:
                max_row_limit = st.slider(
                    f"Max {rank_col_ap} row to consider",
                    min_value=5,
                    max_value=60,
                    value=30,
                    step=1,
                    key="ap_max_row_limit",
                    help="Upper bound for auto row expansion. Lower = cheaper and more focused.",
                )
                allowed_rows_ap = []  # filled after we compute row priorities
            else:
                allowed_rows_ap = st.multiselect(
                    f"Allowed {rank_col_ap} rows (these are your playable 'lines')",
                    options=list(range(1, 61)),
                    default=default_rows,
                    key="ap_allowed_rows",
                )
                allowed_rows_ap = sorted([int(x) for x in allowed_rows_ap]) if allowed_rows_ap else default_rows

        with ap7:
            member_mode_ap = st.selectbox(
                "Member play mode (BOX baseline)",
                [
                    "AUTO: Global member priority (Top1 baseline, then add extra picks by ROI)",
                    "BOX: Play Top1 member pick (per-row)",
                    "BOX: Play Top2 member picks (per-row)",
                    "BOX: Play all 3 family members (per-row)",
                ],
                index=0,
                key="ap_member_mode",
                help="AUTO mode prioritizes members across ALL families by incremental win gain per extra play (so a weak family’s Member#2 can be lower priority than another family’s Member#1).",
            )


        # Straight booster controls (optional)
        sb1, sb2, sb3 = st.columns([1.2, 1.0, 1.0])
        with sb1:
            use_straight_booster = st.checkbox(
                "Enable selective STRAIGHT booster (adds straight only to the highest-confidence pick(s))",
                value=True,
                key="ap_use_straight_booster",
                help="This does NOT add straight to everything. It adds straight only to the top-1 (or top-k) best-confidence ordered pick per day.",
            )
        with sb2:
            max_straights_per_day = st.number_input("Max STRAIGHT picks/day", min_value=0, max_value=50, value=1, step=1, key="ap_max_straights_day")
        with sb3:
            straight_cost = st.number_input("STRAIGHT cost per pick ($)", min_value=0.0, max_value=10.0, value=0.25, step=0.05, key="ap_straight_cost")

        # --- Build evaluation slice ---
        _tmp = out.copy()
        _tmp["Date"] = _parse_date_series(_tmp["Date"])
        _tmp = _tmp.dropna(subset=["Date"])
        max_date = _tmp["Date"].max()
        min_date = max_date - pd.Timedelta(days=int(eval_days) - 1)
        ev = _tmp[_tmp["Date"].between(min_date, max_date)].copy()

        if ev.empty:
            st.warning("No backtest rows in the selected evaluation window.")
        else:
                        # ----------------------------
            # Row (line) priority + auto row expansion (optional)
            # ----------------------------
            fam_mask = ev["ActualFamilyMember"].astype(str).isin(["AABC", "ABBC", "ABCC"])
            pred_mask = ev["Predicted"].astype(bool)

            # If RankPos and BaseScoreRank are identical in this dataset, tell the user (prevents confusion).
            try:
                if ("RankPos" in ev.columns) and ("BaseScoreRank" in ev.columns) and ev["RankPos"].astype(int).equals(ev["BaseScoreRank"].astype(int)):
                    st.info("Heads up: In this backtest export, **RankPos == BaseScoreRank** for all rows, so either basis behaves the same here.")
            except Exception:
                pass

            # Establish candidate rows universe for priority scoring
            _max_row_for_scoring = int(max_row_limit) if 'max_row_limit' in locals() else 60
            _max_row_for_scoring = max(1, min(60, _max_row_for_scoring))
            cand_rows = list(range(1, _max_row_for_scoring + 1))

            # Score each row by "wins per play" using Top1 as the baseline signal (stable + cheapest)
            row_scores = []
            for r in cand_rows:
                rm = ev[rank_col_ap].astype(int).eq(int(r))
                plays = int((fam_mask & pred_mask & rm).sum())
                wins = int((fam_mask & pred_mask & rm & ev.get("MemberHitTop1", False).astype(bool)).sum()) if "MemberHitTop1" in ev.columns else 0
                eff = (wins / plays) if plays > 0 else 0.0
                row_scores.append({"Row": r, "Plays": plays, "WinsTop1": wins, "WinsPerPlayTop1": eff})
            row_scores_df = pd.DataFrame(row_scores).sort_values(["WinsPerPlayTop1", "WinsTop1", "Plays", "Row"], ascending=[False, False, False, True])

            # Choose rows
            if 'auto_rows' in locals() and auto_rows:
                # Add rows in priority order until we meet the target avg BOX wins/day (using the chosen member mode if not AUTO; else Top1 baseline)
                priority_rows = row_scores_df["Row"].astype(int).tolist()
                chosen = []
                # helper to compute avg wins/day under a set of rows and a given member mode
                def _avg_box_wins_per_day(rows_set, mode_name: str) -> tuple[float, float]:
                    if not rows_set:
                        return 0.0, 0.0
                    rm = ev[rank_col_ap].astype(int).isin([int(x) for x in rows_set])
                    base = fam_mask & pred_mask & rm
                    # Box win for a row depends on member picks
                    if mode_name.startswith("BOX: Play Top1"):
                        w = base & ev["MemberHitTop1"].astype(bool)
                        plays_per_row = 1
                    elif mode_name.startswith("BOX: Play Top2"):
                        w = base & ev["MemberHitTop2"].astype(bool)
                        plays_per_row = 2
                    elif mode_name.startswith("BOX: Play all 3"):
                        w = base  # all 3 members guarantees a box hit for any predicted family row
                        plays_per_row = 3
                    else:
                        # AUTO: evaluate baseline (Top1) for row expansion; member expansion handled later
                        w = base & ev["MemberHitTop1"].astype(bool)
                        plays_per_row = 1
                    # day-level wins
                    day_wins = w.groupby(ev["Date"].dt.date).any().astype(int)
                    avg_wins = float(day_wins.mean()) if len(day_wins) else 0.0
                    # plays/day
                    plays_day = (base.groupby(ev["Date"].dt.date).sum() * plays_per_row)
                    avg_plays = float(plays_day.mean()) if len(plays_day) else 0.0
                    return avg_wins, avg_plays

                for r in priority_rows:
                    if r in chosen:
                        continue
                    chosen.append(int(r))
                    avg_w, avg_p = _avg_box_wins_per_day(chosen, member_mode_ap)
                    if avg_w >= float(target_avg_box_wins):
                        break

                allowed_rows_ap = sorted(chosen) if chosen else default_rows
                st.caption(f"Auto rows chosen ({rank_col_ap}): **{', '.join(map(str, allowed_rows_ap))}**")
                with st.expander("Row priority details (why these rows?)", expanded=False):
                    st.dataframe(row_scores_df, use_container_width=True, hide_index=True)
            else:
                # Manual rows already set above
                with st.expander("Row priority details (informational)", expanded=False):
                    st.dataframe(row_scores_df, use_container_width=True, hide_index=True)

            # ----------------------------
            # Build BOX win mask under the selected member mode
            # ----------------------------
            row_mask = ev[rank_col_ap].astype(int).isin([int(x) for x in allowed_rows_ap])

            # Member AUTO mode: treat each additional member pick as its own ROI-ranked action across ALL families
            if member_mode_ap.startswith("AUTO"):
                base_rows = fam_mask & pred_mask & row_mask
                # Baseline = Top1 for all predicted family rows
                base_win = base_rows & ev["MemberHitTop1"].astype(bool)
                # day-level wins under baseline
                day_any = base_win.groupby(ev["Date"].dt.date).any()
                win_days = set(day_any[day_any].index.tolist())
                all_days = sorted(ev["Date"].dt.date.unique().tolist())
                target_wins_total = float(target_avg_box_wins) * max(1, len(all_days))

                # Build candidate extra picks:
                # - Add Top2 when Top1 missed but Top2 would hit.
                # - Add 3rd member only when both Top1 and Top2 missed (meaning actual is the 3rd).
                cand = []
                if "MemberHitTop2" in ev.columns:
                    # Top2 incremental win rows
                    top2_inc = base_rows & (~ev["MemberHitTop1"].astype(bool)) & (ev["MemberHitTop2"].astype(bool))
                    for d, sub in ev[top2_inc].groupby(ev["Date"].dt.date):
                        # one extra pick can win that specific row; but we care if it converts the day
                        cand.append({"kind":"TOP2", "date": d, "gain_day": (d not in win_days)})
                # 3rd member incremental (only if we allow it; we will allow if still short after Top2)
                third_inc = base_rows & (~ev["MemberHitTop2"].astype(bool)) & (~ev["MemberHitTop1"].astype(bool))
                for d, sub in ev[third_inc].groupby(ev["Date"].dt.date):
                    cand.append({"kind":"THIRD", "date": d, "gain_day": (d not in win_days)})

                # Greedy add extra picks that convert currently losing days first
                extra_picks = []
                # We approximate 1 candidate per date for simplicity (best possible), because we only need daily coverage and extra pick converts the day.
                # Prioritize TOP2 over THIRD (cheaper confidence-wise) when both could convert the day.
                for kind in ["TOP2", "THIRD"]:
                    for item in [c for c in cand if c["kind"]==kind]:
                        if len(win_days) >= target_wins_total:
                            break
                        if item["date"] in win_days:
                            continue
                        win_days.add(item["date"])
                        extra_picks.append(item)
                    if len(win_days) >= target_wins_total:
                        break

                # Construct final win mask:
                # baseline wins + any wins due to extra picks (we approximate as converting the day when possible)
                ev["_BoxWinBaselineTop1"] = base_win
                # For reporting: effective plays per row = 1 baseline, plus extra picks counted at day-level
                ev["BoxWin"] = base_win  # row-level; day-level conversion handled below
                members_per_row = 1  # baseline
                auto_extra_picks = len(extra_picks)
            else:
                auto_extra_picks = 0
                if member_mode_ap.startswith("BOX: Play Top1"):
                    win_mask = fam_mask & pred_mask & row_mask & ev["MemberHitTop1"].astype(bool)
                    members_per_row = 1
                elif member_mode_ap.startswith("BOX: Play Top2"):
                    win_mask = fam_mask & pred_mask & row_mask & ev["MemberHitTop2"].astype(bool)
                    members_per_row = 2
                else:
                    win_mask = fam_mask & pred_mask & row_mask
                    members_per_row = 3
                ev["BoxWin"] = win_mask

            # If AUTO member mode, compute a day-level win series directly and keep it for downstream stats.
            if member_mode_ap.startswith("AUTO"):
                # baseline day wins
                day_wins_series = ev["_BoxWinBaselineTop1"].groupby(ev["Date"].dt.date).any().astype(int)
                # add extra picks as day-level conversions
                for it in extra_picks:
                    if it["date"] in day_wins_series.index:
                        day_wins_series.loc[it["date"]] = 1
                ev["_BoxWinDay"] = ev["Date"].dt.date.map(lambda d: bool(day_wins_series.get(d, 0)))
            else:
                ev["_BoxWinDay"] = ev["Date"].dt.date.map(lambda d: bool((ev.loc[ev["Date"].dt.date==d, "BoxWin"]).any()))


# Plays/day and wins/day per core
            cores = sorted(ev["Core"].astype(str).unique().tolist())
            days = sorted(ev["Date"].dt.date.unique().tolist())

            # Per-core daily metrics
            per_core = []
            for c in cores:
                g = ev[ev["Core"].astype(str) == str(c)].copy()
                # BOX plays = predicted rows within allowed lines * members_per_row
                plays = int((g[g["Predicted"].astype(bool) & g[rank_col_ap].astype(int).isin([int(x) for x in allowed_rows_ap])].shape[0]) * members_per_row) if (not g.empty) else 0
                # But plays should be per day average, so compute properly:
                plays_by_day = (
                    g[g["Predicted"].astype(bool) & g[rank_col_ap].astype(int).isin([int(x) for x in allowed_rows_ap])]
                    .groupby(g["Date"].dt.date)
                    .size()
                    .reindex(days, fill_value=0)
                    .astype(int)
                    * int(members_per_row)
                )
                wins_by_day = (
                    g[g["BoxWin"]]
                    .groupby(g["Date"].dt.date)
                    .size()
                    .reindex(days, fill_value=0)
                    .astype(int)
                )
                # Predictability signals (member accuracy), using the same evaluation slice
                top1_rate = float(g["MemberHitTop1"].astype(bool).mean()) if "MemberHitTop1" in g.columns else np.nan
                top2_rate = float(g["MemberHitTop2"].astype(bool).mean()) if "MemberHitTop2" in g.columns else np.nan

                # Simple "likely winner" signal (data-driven but conservative):
                # - prioritize small row numbers, more recent activity, and strong Top1 member accuracy
                avg_rank = float(g[rank_col_ap].astype(int).mean()) if (rank_col_ap in g.columns and len(g)) else np.nan
                avg_hitsw = float(g["HitsWindow"].astype(float).mean()) if ("HitsWindow" in g.columns and len(g)) else np.nan
                avg_dslh = float(g["DaysSinceLastHit"].astype(float).mean()) if ("DaysSinceLastHit" in g.columns and len(g)) else np.nan
                if (not np.isnan(top1_rate)) and (not np.isnan(avg_rank)) and (not np.isnan(avg_dslh)):
                    if (avg_rank <= 5) and (top1_rate >= 0.45) and (avg_dslh <= 30):
                        signal = "🟢 High"
                    elif (avg_rank <= 12) and (top1_rate >= 0.35) and (avg_dslh <= 60):
                        signal = "🟡 Medium"
                    else:
                        signal = "🔴 Low"
                else:
                    signal = "—"

                per_core.append(
                    {
                        "Core": str(c).zfill(3),
                        "EvalDays": len(days),
                        "BoxWinDays": int((wins_by_day > 0).sum()),
                        "BoxWinsTotal": int(wins_by_day.sum()),
                        "AvgBoxWinsPerDay": float(wins_by_day.sum() / max(1, len(days))),
                        "AvgBoxPlaysPerDay": float(plays_by_day.mean()),
                        "AvgRow": round(avg_rank, 2) if not np.isnan(avg_rank) else np.nan,
                        "Signal": signal,
                        "Top1Rate": round(top1_rate, 4) if not np.isnan(top1_rate) else np.nan,
                        "Top2Rate": round(top2_rate, 4) if not np.isnan(top2_rate) else np.nan,
                    }
                )

            per_core_df = pd.DataFrame(per_core)
            per_core_df["EffWinsPerPlay"] = (
                per_core_df["AvgBoxWinsPerDay"] / per_core_df["AvgBoxPlaysPerDay"].replace({0: np.nan})
            )
            per_core_df["EffWinsPerPlay"] = per_core_df["EffWinsPerPlay"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            st.markdown("**Core cadence + predictability snapshot (evaluation window)**")
            st.dataframe(
                per_core_df.sort_values(["AvgBoxWinsPerDay", "EffWinsPerPlay"], ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            # --- Greedy selection: minimum cores to reach target avg wins/day under play cap ---
            st.markdown("**Auto-select minimum cores to reach target wins/day**")
            st.caption(
                "This uses a greedy set-building approach based on the *marginal increase* in wins/day. "
                "It is real-data-driven from the backtest rows you already generated."
            )

            # Precompute per-core wins_by_day and plays_by_day for fast greedy union
            core_wins = {}
            core_plays = {}
            for c in cores:
                g = ev[ev["Core"].astype(str) == str(c)].copy()
                core_wins[str(c).zfill(3)] = (
                    g[g["BoxWin"]].groupby(g["Date"].dt.date).size().reindex(days, fill_value=0).astype(int).values
                )
                core_plays[str(c).zfill(3)] = (
                    g[g["Predicted"].astype(bool) & g[rank_col_ap].astype(int).isin([int(x) for x in allowed_rows_ap])].groupby(g["Date"].dt.date).size().reindex(days, fill_value=0).astype(int).values
                    * int(members_per_row)
                )

            selected = []
            total_wins_vec = np.zeros(len(days), dtype=int)
            total_plays_vec = np.zeros(len(days), dtype=int)

            remaining = [str(c).zfill(3) for c in cores]
            # Sort candidates by efficiency first to make greedy stable
            remaining = sorted(
                remaining,
                key=lambda x: (
                    float(per_core_df.loc[per_core_df["Core"] == x, "EffWinsPerPlay"].values[0])
                    if (per_core_df["Core"] == x).any() else 0.0
                ),
                reverse=True,
            )

            def _avg(v):
                return float(np.sum(v) / max(1, len(days)))

            target_reached = False
            for _ in range(0, 60):  # safety cap
                cur_avg_wins = _avg(total_wins_vec)
                cur_avg_plays = float(np.mean(total_plays_vec))
                if (cur_avg_wins >= float(target_avg_box_wins)) and (cur_avg_plays <= float(max_box_plays_per_day)):
                    target_reached = True
                    break

                best = None
                best_gain = -1.0
                best_new_plays = None

                for c in remaining:
                    new_wins = total_wins_vec + core_wins[c]
                    new_plays = total_plays_vec + core_plays[c]
                    gain = _avg(new_wins) - _avg(total_wins_vec)
                    # Soft enforce play cap by penalizing cores that push plays too high
                    plays_penalty = max(0.0, (float(np.mean(new_plays)) - float(max_box_plays_per_day)) / max(1.0, float(max_box_plays_per_day)))
                    score = gain - (0.5 * plays_penalty)
                    if score > best_gain:
                        best_gain = score
                        best = c
                        best_new_plays = new_plays

                if best is None:
                    break

                selected.append(best)
                total_wins_vec = total_wins_vec + core_wins[best]
                total_plays_vec = total_plays_vec + core_plays[best]
                remaining = [c for c in remaining if c != best]

                if not remaining:
                    break

            # Report result
            avg_wins = _avg(total_wins_vec)
            avg_plays = float(np.mean(total_plays_vec))
            avg_cost = avg_plays * float(cost_per_play_ap)

            st.write(
                f"**Selected cores:** {', '.join(selected) if selected else '(none)'}"
            )
            st.write(
                f"**Expected avg BOX wins/day (eval window):** {avg_wins:.3f}  |  "
                f"**Avg BOX plays/day:** {avg_plays:.1f}  |  "
                f"**Avg BOX cost/day:** ${avg_cost:.2f}"
            )

            if avg_wins < float(target_avg_box_wins):
                st.warning(
                    "Even after selecting many cores, the evaluation slice did not reach your target avg wins/day under the current row + member settings. "
                    "Try widening allowed rows, switching to Top2/all-3 members, or increasing the play cap."
                )

            # --- Straight booster: build a daily top-1 ordered pick (real, walk-forward safe) ---
            if use_straight_booster and (max_straights_per_day > 0) and selected:
                st.markdown("**Selective STRAIGHT booster (daily top pick)**")
                st.caption(
                    "This uses a simple, **walk-forward safe** rule for ordering: for the chosen BOX member, "
                    "predict the STRAIGHT as LAST(stream exact hit for that member) → else MODE(stream exact hit) → else skip. "
                    "It is meant as a conservative booster, not a guarantee."
                )

                # Build a tiny per-day straight suggestion list based on the selected cores and predicted rows
                # Note: We only suggest straights on rows where you are already playing a BOX pick.
                ev_sel = ev[ev["Core"].astype(str).isin(selected)].copy()
                ev_sel = ev_sel[pred_mask & row_mask].copy()
                ev_sel["DateOnly"] = ev_sel["Date"].dt.date

                # Helper: infer the chosen member label for the row under current member mode
                def _row_chosen_member_label(r):
                    if member_mode_ap.startswith("BOX: Play Top1"):
                        return str(_safe_get(r,"PredMemberTop1", "")) if pd.notna(_safe_get(r,"PredMemberTop1")) else ""
                    if member_mode_ap.startswith("BOX: Play Top2"):
                        # We'll still choose the top1 label for straight ordering (cheapest & most consistent)
                        return str(_safe_get(r,"PredMemberTop1", "")) if pd.notna(_safe_get(r,"PredMemberTop1")) else ""
                    # all-3: choose the most likely member by Top1 prediction if present, else blank
                    return str(_safe_get(r,"PredMemberTop1", "")) if pd.notna(_safe_get(r,"PredMemberTop1")) else ""

                ev_sel["ChosenMemberLabel"] = ev_sel.apply(_row_chosen_member_label, axis=1)

                # Map (core, member_label) -> 4-digit BOX number (canonical member string)
                # We reuse members_from_core which returns box-keys (sorted digits), then pick the matching member label.
                def _member_box_for_label(core_key: str, lab: str) -> str | None:
                    core_key = canonical_core_key(core_key)
                    fam_boxes = members_from_core(core_key, "AABC")
                    if lab == "AABC":
                        return str(fam_boxes[0])
                    if lab == "ABBC":
                        return str(fam_boxes[1])
                    if lab == "ABCC":
                        return str(fam_boxes[2])
                    return None

                # Build straight suggestion per day: pick the row with best member predictability (Top1Rate proxy)
                # Use the per_core_df Top1Rate as a cheap, real-data confidence signal.
                top1_rate_map = {
                    str(r["Core"]).zfill(3): float(r["Top1Rate"]) if pd.notna(r["Top1Rate"]) else 0.0
                    for _, r in per_core_df.iterrows()
                }

                sugg_rows = []
                for d in days:
                    day_rows = ev_sel[ev_sel["DateOnly"] == d].copy()
                    if day_rows.empty:
                        continue
                    # choose up to k rows by confidence
                    day_rows["CoreZ"] = day_rows["Core"].astype(str).apply(lambda x: str(x).zfill(3))
                    day_rows["Conf"] = day_rows["CoreZ"].map(top1_rate_map).fillna(0.0)
                    day_rows = day_rows.sort_values(["Conf", rank_col_ap], ascending=[False, True]).head(int(max_straights_per_day))
                    for _, r in day_rows.iterrows():
                        corez = str(r["CoreZ"])
                        lab = str(_safe_get(r,"ChosenMemberLabel",""))
                        boxnum = _member_box_for_label(corez, lab)
                        if not boxnum:
                            continue
                        # Determine the straight guess from train history (Date < d) within this stream
                        td = pd.to_datetime(d)
                        stream = str(_safe_get(r,"Stream",""))
                        train = df_all.copy()
                        train["Date"] = _parse_date_series(train["Date"])
                        train = train.dropna(subset=["Date"])
                        train = train[train["Date"] < td].copy()
                        if train.empty:
                            continue
                        # same stream exact hits for this member
                        train_s = train[train["Stream"].astype(str) == stream].copy() if "Stream" in train.columns else train.copy()
                        train_s["Result4"] = train_s["Result"].apply(lambda x: extract_4digit(x) or "")
                        train_s = train_s[train_s["Result4"].str.len() == 4].copy()
                        # filter to this member by box_key match
                        train_s["BoxKey"] = train_s["Result4"].apply(box_key)
                        mk = box_key(boxnum)
                        train_m = train_s[train_s["BoxKey"] == mk].copy()
                        straight_pick = None
                        if not train_m.empty:
                            # LAST(stream)
                            train_m = train_m.sort_values("Date")
                            straight_pick = str(train_m["Result4"].iloc[-1])
                        else:
                            # MODE(stream) over the member
                            # (If none in stream, fall back to global)
                            train_g = train.copy()
                            train_g["Result4"] = train_g["Result"].apply(lambda x: extract_4digit(x) or "")
                            train_g = train_g[train_g["Result4"].str.len() == 4].copy()
                            train_g["BoxKey"] = train_g["Result4"].apply(box_key)
                            train_mg = train_g[train_g["BoxKey"] == mk].copy()
                            if not train_mg.empty:
                                straight_pick = str(train_mg["Result4"].value_counts().idxmax())

                        if straight_pick:
                            sugg_rows.append(
                                {
                                    "Date": str(d),
                                    "Stream": stream,
                                    "Core": corez,
                                    "MemberLabel": lab,
                                    "BoxNumber": boxnum,
                                    "StraightPick": straight_pick,
                                    "StraightCost": float(straight_cost),
                                    "BoxCost": float(cost_per_play_ap),
                                }
                            )

                sugg_df = pd.DataFrame(sugg_rows)
                if sugg_df.empty:
                    st.info("No straight booster picks could be generated for the evaluation window under the current settings.")
                else:
                    st.dataframe(sugg_df.head(200), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download straight booster picks (CSV)",
                        data=sugg_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"straight_booster_picks_last{int(eval_days)}d.csv",
                        mime="text/csv",
                        key="ap_dl_straight_booster",
                    )

        st.caption(
            "Note: This planner is **evaluation-window based**. Once you like the settings, we will wire the same core+row+member rules into the daily 'Play Today' outputs."
        )

# ----------------------------
# Baseline disk cache (optional)
# ----------------------------
def _baseline_paths(core: str, window_days: int):
    core = str(core).zfill(3)
    base = DISK_CACHE_DIR / f"baseline_{window_days}d_{core}"
    return {
        "stream": base.with_suffix(".stream.parquet"),
        "pos": base.with_suffix(".pos.parquet"),
        "meta": base.with_suffix(".meta.json"),
    }

def _load_baseline_from_disk(core: str, window_days: int, expected_last_date: str | None):
    p = _baseline_paths(core, window_days)
    meta = _read_meta(p["meta"])
    if not meta:
        return None, None, None
    if expected_last_date and _safe_get(meta,"last_date") != expected_last_date:
        return None, None, meta
    stream_df = _safe_read_table(p["stream"])
    pos_df = _safe_read_table(p["pos"])
    if stream_df is None or pos_df is None:
        return None, None, meta
    return stream_df, pos_df, meta

def _save_baseline_to_disk(core: str, window_days: int, stream_df, pos_df, last_date: str | None):
    p = _baseline_paths(core, window_days)
    _safe_write_table(stream_df, p["stream"])
    _safe_write_table(pos_df, p["pos"])
    _write_meta(p["meta"], {
        "core": str(core).zfill(3),
        "window_days": int(window_days),
        "last_date": last_date,
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def get_stream_stats_cached(core: str, window_days: int, df_all, last_date: str | None):
    key = f"stream_stats::{window_days}::{str(core).zfill(3)}"
    if key in st.session_state:
        return st.session_state[key]
    # Try disk cache
    stream_df, pos_df, meta = _load_baseline_from_disk(core, window_days, expected_last_date=last_date)
    if stream_df is not None:
        st.session_state[key] = stream_df
        # also stash pos if present
        if pos_df is not None:
            st.session_state[f"pos_map::{window_days}::{str(core).zfill(3)}"] = pos_df
        return stream_df
    # Compute fresh
    hits = compute_core_hits(df_all, core, structures=("AABC",))
    stream_df = stream_summary(df_all, hits, window_days=window_days)
    st.session_state[key] = stream_df
    return stream_df


def compute_stream_stats(df_all: pd.DataFrame, core: str, window_days: int | None = None, exclude_md: bool = False) -> pd.DataFrame:
    """Back-compat wrapper used by the Northern Lights block."""
    if window_days is None:
        window_days = 180
    # exclude_md is already applied upstream; kept only for compatibility
    last_date = most_recent_date(df_all)
    last_s = None
    if last_date is not None and not pd.isna(last_date):
        try:
            last_s = str(pd.to_datetime(last_date).date())
        except Exception:
            last_s = None
    return get_stream_stats_cached(core=str(core), window_days=int(window_days), df_all=df_all, last_date=last_s)


def build_northern_star_buckets(
    stats_df: pd.DataFrame,
    stream: str,
    top_n: int = 12,
    due_ranks: Tuple[int, int] = (13, 60),
    seed_core_key: str = "core",
    include_24h: bool = True,
    df_24: Optional[pd.DataFrame] = None,
    core: str = "000",
    **kwargs,
) -> Dict[str, object]:
    """
    Back-compat bucket logic for the Northern Lights table:
    - Base bucket: Top N streams by HitsPerWeek
    - Due bucket: from base ranks [due_from..due_to], take Top cfg.top_due by DaysSinceLastHit
    Returns fields expected by the Northern Lights renderer.
    """
    if stats_df is None or stats_df.empty:
        return {}

    s = stats_df.copy()
    s = s.sort_values(["HitsPerWeek", "HitsWindow"], ascending=[False, False]).reset_index(drop=True)
    s["BaseRank"] = s.index + 1

    # Base top
    base_top_streams = set(s.head(int(top_n))["Stream"].astype(str).tolist())

    # Due candidates are chosen from the *base-ranked* band
    d1, d2 = int(due_ranks[0]), int(due_ranks[1])
    band = s[(s["BaseRank"] >= d1) & (s["BaseRank"] <= d2)].copy()
    if band.empty:
        due_top_streams = set()
    else:
        band = band.sort_values(["DaysSinceLastHit", "HitsPerWeek"], ascending=[False, False])
        due_top_streams = set(band.head(int(getattr(st.session_state.get("_cfg", RankConfig()), "top_due", 8)))["Stream"].astype(str).tolist())

    in_base = str(stream) in base_top_streams
    in_due = str(stream) in due_top_streams

    # Pull row for this stream
    row = s[s["Stream"].astype(str) == str(stream)]
    if row.empty:
        return {}
    r = row.iloc[0]
    hits = _safe_int(_safe_get(r,"HitsWindow", 0)) or 0
    hpw = float(_safe_get(r,"HitsPerWeek", 0.0) or 0.0)
    dslh = (_safe_int(_safe_get(r,"DaysSinceLastHit", 0)) or 0)

    # Due pressure as soft signal
    due_pressure = 0.0
    if in_due:
        due_pressure = float(dslh)

    # Optional 24h soft signal: add a small nudge if this core hit this stream in df_24
    if include_24h and df_24 is not None and not df_24.empty:
        try:
            cache_key = f"_nl_24h_corehits_{core}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = compute_core_hits(df_24, str(core), structures=["AABC"])
            df_24h_core_hits = st.session_state.get(cache_key, pd.DataFrame())
            if not df_24h_core_hits.empty and (df_24h_core_hits["Stream"].astype(str) == str(stream)).any():
                due_pressure += 1.0
        except Exception:
            pass

    seed_key = canonical_core_key(str(core))

    bucket_label = "Top12" if in_base else ("Due" if in_due else "")
    bucket_pick = "BASE" if in_base else ("DUE" if in_due else "")

    return {
        "Top12": bucket_label,
        "BucketPick": bucket_pick,
        "SeedKey": seed_key,
        "DuePressure": due_pressure,
        "HitsPerWeek": hpw,
        "Hits": hits,
        "DaysSinceLastHit": dslh,
    }




def get_pos_map_cached(core: str, window_days: int, stream_stats_df, last_date: str | None):
    k = f"pos_map::{window_days}::{str(core).zfill(3)}"
    if k in st.session_state:
        return st.session_state[k]
    # If stream_stats was loaded from disk, pos may already be in session_state
    pos_df, _ = position_percentile_map(stream_stats_df)
    st.session_state[k] = pos_df
    # Save to disk alongside stream stats (best effort)
    try:
        _save_baseline_to_disk(core, window_days, stream_stats_df, pos_df, last_date)
    except Exception:
        pass
    return pos_df

# Baseline cache builder UI (runs only after history is loaded)
st.subheader("Baseline cache builder")
st.caption("This aggregates the Northern Star bucket picks across your selected cores and ranks them using a universal score (recent strength + due pressure + position-percentile strength).")

build_both = st.checkbox("Build cache for both windows (180 & 365)", value=False)
if st.button("Build baseline cache now"):
    if df_all is None or df_all.empty:
        st.warning("Upload a history file first.")
    else:
        build_windows = [180, 365] if build_both else [window_days]
        built = 0
        for w in build_windows:
            for c in cores_for_cache:
                stream_df = get_stream_stats_cached(c, w, df_all=df_all, last_date=last_all)
                pos_df = get_pos_map_cached(c, w, stream_df, last_date=last_all)
                try:
                    _save_baseline_to_disk(c, w, stream_df, pos_df, last_all)
                    built += 1
                except Exception:
                    pass
        st.success(f"Cache built for {built} core-window combinations. Latest history date: {last_all}")
        # Also build a unified (Stream, Core) ranking table so other tabs (e.g., Play Today)
        # can function immediately after cache build.
        try:
            rank_frames = []
            for w in build_windows:
                rows = []
                for c in cores_for_cache:
                    try:
                        sdf = get_stream_stats_cached(c, w, df_all=df_all, last_date=last_all)
                        if sdf is None or getattr(sdf, 'empty', False):
                            continue
                        tmp = sdf.copy()
                        tmp['Core'] = c
                        rows.append(tmp)
                    except Exception:
                        continue
                if rows:
                    df_rank = pd.concat(rows, ignore_index=True)
                    # pick a score column consistent with the app's existing engine
                    score_col = None
                    for cand in ['TotalScore', 'Score', 'FinalScore', 'BaseScore', 'Base_Score', 'RankScore', 'NS_Score']:
                        if cand in df_rank.columns:
                            score_col = cand
                            break
                    if score_col is None:
                        # fallback: most frequent hits in-window if present
                        for cand in ['Hits', 'HitCount', 'count', 'Freq', 'Frequency']:
                            if cand in df_rank.columns:
                                score_col = cand
                                break
                    if score_col is not None:
                        df_rank = df_rank.sort_values(score_col, ascending=False)
                    st.session_state[f'ns_stream_rank_df_{w}'] = df_rank
                    _save_ns_stream_rank_to_disk(int(w), df_rank)
                    rank_frames.append(df_rank.assign(_window=w))

            # default = current window if available
            if len(build_windows) == 1 and f'ns_stream_rank_df_{build_windows[0]}' in st.session_state:
                st.session_state['ns_stream_rank_df'] = st.session_state[f'ns_stream_rank_df_{build_windows[0]}']
            elif f'ns_stream_rank_df_{window_days}' in st.session_state:
                st.session_state['ns_stream_rank_df'] = st.session_state[f'ns_stream_rank_df_{window_days}']
            elif rank_frames:
                st.session_state['ns_stream_rank_df'] = pd.concat(rank_frames, ignore_index=True)
        except Exception:
            pass





# ===============================
# Main tabs
# ===============================

tab_labels = ["Northern Star (v51)", "Northern Lights (Master playlist)", "Core view", "Backtest (optional)", "7-Core Execution (Play Today)", "Validator (Walk-forward Top2)"]
tabs = st.tabs(tab_labels)
_t_ns = tabs[0]
_t_nl = tabs[1]
_t_core = tabs[2]
_t_bt = tabs[3]
_t_exec = tabs[4]
_t_val = tabs[5]

# --- Northern Lights master playlist (best -> worst across streams/cores) ---
if _t_nl is None:
    _t_nl = st.container()
with _t_nl:
    st.subheader("Northern Lights master playlist")
    st.caption("Aggregated bucket picks across your selected cores. Use this as your universal stream playlist.")
    # Optional performance toggle: build Northern Lights across ALL tracked cores (ignores selection)
    nl_use_all_cores = st.checkbox(
        "Use ALL tracked cores for Northern Lights (can be slower)",
        value=bool(st.session_state.get("_nl_use_all_cores", False)),
        key="_nl_use_all_cores",
        help="If enabled, the playlist ranks every stream using every tracked core. This can be slower unless baseline cache exists.",
    )


    # Ensure cores_for_cache is always defined (selected cores for cache building / views)
    cores_for_cache = list(st.session_state.get('cores_for_cache') or st.session_state.get('selected_cores') or [])
    if not cores_for_cache:
        cores_for_cache = [core_for_view] if 'core_for_view' in locals() else (cores[:1] if 'cores' in locals() and cores else ['000'])
    cores_for_cache = [str(c).zfill(3) for c in cores_for_cache]
    # If requested, ignore selection and use the full tracked core list.
    if nl_use_all_cores:
        cores_for_cache = [str(c).zfill(3) for c in CORE_PRESETS]
        st.info(f"Northern Lights is using ALL tracked cores ({len(cores_for_cache)}). If this feels slow, build baseline cache first in Cache Builder.")

    if not cores_for_cache:
        st.info("Select one or more cores above to populate the playlist.")
    else:
        # NOTE: Northern Lights playlist computation can be expensive; defer by default for faster app load.
        _nl_run_now = st.button("Build / Refresh Northern Lights master playlist")
        _nl_auto = st.checkbox("Auto-build Northern Lights playlist each rerun (slower)", value=False)
        if not (_nl_run_now or _nl_auto):
            st.info("Northern Lights playlist build is deferred. Click **Build / Refresh Northern Lights master playlist** to compute it.")
        else:
            cfg = st.session_state.get("_cfg", RankConfig())
            include_24h = bool(st.session_state.get("include_24h", True))
    
            # Build a master list: (core, stream) -> universal score
            # v51: In ALL-CORES mode, enforce strict cache-only for safety/performance.
            stats_by_core: Dict[str, pd.DataFrame] = {}
            if nl_use_all_cores:
                expected_last = last_all if isinstance(last_all, str) else None
                missing = []
                for _c in cores_for_cache:
                    ss, _pos_df, _meta = _load_baseline_from_disk(_c, cfg.window_days, expected_last_date=expected_last)
                    if ss is None or ss.empty:
                        missing.append(_c)
                    else:
                        stats_by_core[_c] = ss
                if missing:
                    st.error("ALL-CORES mode is cache-only. Missing baseline caches for: " + ", ".join(missing))
                    st.caption("Build caches in the Cache Builder section, then rerun.")
                    st.stop()
    
            rows = []
            ns_stream_rows = []  # collect per-core stream tables for downstream 'Play Today' tools
            for core in cores_for_cache:
                try:
                    stats_df = stats_by_core.get(core) if nl_use_all_cores else compute_stream_stats(df_all, core, window_days=window_days, exclude_md=False)
                except Exception:
                    stats_df = pd.DataFrame()
    
                if stats_df is None or stats_df.empty:
                    continue
    
                # Save this per-core stream ranking table for other panels (e.g., optimizer)
                try:
                    _tmp_ns = stats_df.copy()
                    _tmp_ns["core"] = canonical_core_key(str(core))
                    ns_stream_rows.append(_tmp_ns)
                except Exception:
                    pass
    
                for stream in stats_df["Stream"].astype(str).tolist():
                    meta = build_northern_star_buckets(
                        stats_df=stats_df,
                        stream=stream,
                        top_n=cfg.top_base,
                        due_ranks=(cfg.due_from_rank, cfg.due_to_rank),
                        seed_core_key=canonical_core_key(str(core)),
                        include_24h=include_24h,
                        df_24=df_24h,
                        core=str(core),
                    )
                    if not meta:
                        continue
    
                    # UniversalScore is what we rank by in the playlist
                    # (recent strength + due pressure + position-percentile strength)
                    # Position-percentile strength comes from the cached pos map
                    try:
                        # Per-stream RankPos percentile map (cache-backed)
                        last_s = None
                        try:
                            last_s = most_recent_date_for_stream(df_all, stream)
                        except Exception:
                            last_s = None
    
                        pos_df = get_pos_map_cached(str(core), int(window_days), stats_df, last_date=last_s)
    
                        # Position strength (by RankPos)
                        try:
                            rankpos = int(_safe_get(meta,"RankPos", 9999) or 9999)
                        except Exception:
                            rankpos = 9999
                        p = 0.0
                        if pos_df is not None and not pos_df.empty and "RankPos" in pos_df.columns:
                            try:
                                _strength_col = "PctStrength" if "PctStrength" in pos_df.columns else ("HitCountPctile" if "HitCountPctile" in pos_df.columns else None)
                                if _strength_col:
                                    _m = pos_df[pos_df["RankPos"].astype(int) == int(rankpos)]
                                    if not _m.empty:
                                        p = float(_m.iloc[0][_strength_col])
                            except Exception:
                                p = 0.0
    
                        # Base signals from bucket meta
                        hits_pw = float(_safe_get(meta,"HitsPerWeek", 0.0) or 0.0)
                        days_since = float(_safe_get(meta,"DaysSinceLastHit", 0.0) or 0.0)
                        due_bucket_pressure = float(_safe_get(meta,"DuePressure", 0.0) or 0.0)
    
                        # Seed Traits score (soft)
                        seed = _prev_seed_by_stream.get(str(stream))
                        seed_score = 0.0
                        if st.session_state.get("enable_seed_traits", True) and seed_traits_pos_lookup:
                            seed_score, _seed_matches = compute_seed_traits_score(
                                str(core),
                                seed,
                                str(stream),
                                pos_lookup=seed_traits_pos_lookup,
                                neg_lookup=seed_traits_neg_lookup,
                                last5_union_digits_by_stream=_last5_union_by_stream,
                            )
    
                        # Cadence score (soft) — mean gap baseline from window hits
                        try:
                            _total_hits = float(stats_df["HitsWindow"].sum()) if "HitsWindow" in stats_df.columns else 0.0
                        except Exception:
                            _total_hits = 0.0
                        mean_gap_days = (window_days / _total_hits) if _total_hits > 0 else 0.0
                        cadence_score = (
                            compute_cadence_score(days_since, mean_gap_days)
                            if (st.session_state.get("enable_cadence", True) and mean_gap_days > 0)
                            else 0.0
                        )
    
                        # Universal score (soft additive; no eliminations)
                        due_w = float(st.session_state.get("due_weight", 0.20))
                        pos_w = float(st.session_state.get("pos_weight", 0.25))
                        st_w = float(st.session_state.get("seed_traits_weight", 0.35))
                        cad_w = float(st.session_state.get("cadence_weight", 0.25))
    
                        universal = (
                            hits_pw
                            + (min(days_since, 50.0) * 0.01 * due_w)
                            + (p * 0.01 * pos_w)
                            + (seed_score * st_w if st.session_state.get("enable_seed_traits", True) else 0.0)
                            + (cadence_score * cad_w if st.session_state.get("enable_cadence", True) else 0.0)
                        )
    
                        rows.append({
                            "Core": str(core),
                            "Stream": str(stream),
                            "BucketPick": str(_safe_get(meta,"BucketPick", "")),
                            "UniversalScore": float(universal),
                            "HitsPerWeek": float(hits_pw),
                            "DaysSinceLastHit": float(days_since),
                            "DueBucketPressure": float(due_bucket_pressure),
                            "DuePressure": float(due_bucket_pressure),
                            "PctStrength": float(p),
                            "Seed": seed,
                            "SeedTraitsScore": float(seed_score),
                            "CadenceScore": float(cadence_score),
                            "TriggerBoost": float(_safe_get(meta,"TriggerBoost", 0.0) or 0.0),
                            "Hits": float(_safe_get(meta,"Hits", 0.0) or 0.0),
                            "RankPos": int(rankpos) if isinstance(rankpos, int) else int(_safe_get(meta,"RankPos", 9999) or 9999),
                            "BaseScore": float(_safe_get(meta,"BaseScore", 0.0) or 0.0),
                            "DueIndex": float(_safe_get(meta,"DueIndex", 0.0) or 0.0),
                        })
                    except Exception:
                        # Fallback: still emit a row without the position/traits features
                        try:
                            hits_pw = float(_safe_get(meta,"HitsPerWeek", 0.0) or 0.0)
                        except Exception:
                            hits_pw = 0.0
                        try:
                            days_since = float(_safe_get(meta,"DaysSinceLastHit", 0.0) or 0.0)
                        except Exception:
                            days_since = 0.0
                        rows.append({
                            "Core": str(core),
                            "Stream": str(stream),
                            "BucketPick": str(_safe_get(meta,"BucketPick", "")),
                            "UniversalScore": float(hits_pw),
                            "HitsPerWeek": float(hits_pw),
                            "DaysSinceLastHit": float(days_since),
                            "DueBucketPressure": float(_safe_get(meta,"DuePressure", _safe_get(meta,"DueBucketPressure", 0.0)) or 0.0),
                            "DuePressure": float(_safe_get(meta,"DuePressure", _safe_get(meta,"DueBucketPressure", 0.0)) or 0.0),
                            "PctStrength": 0.0,
                            "Seed": _prev_seed_by_stream.get(str(stream)),
                            "SeedTraitsScore": 0.0,
                            "CadenceScore": 0.0,
                            "TriggerBoost": float(_safe_get(meta,"TriggerBoost", 0.0) or 0.0),
                            "Hits": float(_safe_get(meta,"Hits", 0.0) or 0.0),
                            "RankPos": int(_safe_get(meta,"RankPos", 9999) or 9999),
                            "BaseScore": float(_safe_get(meta,"BaseScore", 0.0) or 0.0),
                            "DueIndex": float(_safe_get(meta,"DueIndex", 0.0) or 0.0),
                        })
    
            if not rows:
                st.warning("No playlist rows were produced. Double-check that your history file contains your selected cores in AABC structure.")
            else:
                nl_df = pd.DataFrame(rows)
                # Cache the per-core stream ranking tables so other panels can reuse them without recompute
                try:
                    if ns_stream_rows:
                        st.session_state["ns_stream_rank_df"] = pd.concat(ns_stream_rows, ignore_index=True)
                    else:
                        # Fallback: derive a reasonable (Stream, Core) ranking table from the already-built Northern Lights playlist,
                        # so downstream panels (like 7-Core Execution) never fail just because the optional NS stream table had no rows.
                        if "nl_df" in locals() and isinstance(nl_df, pd.DataFrame) and (not nl_df.empty):
                            cols = [c for c in ["Stream", "Core", "BaseScore", "RankPos", "Score", "TotalScore"] if c in nl_df.columns]
                            if ("Stream" in cols) and ("Core" in cols):
                                tmp = nl_df[cols].copy()
                                # Pick a score column for sorting
                                score_col = None
                                for sc in ["BaseScore", "TotalScore", "Score", "RankPos"]:
                                    if sc in tmp.columns:
                                        score_col = sc
                                        break
                                if score_col is None:
                                    tmp["_score"] = 0.0
                                    score_col = "_score"
                                tmp = tmp.sort_values(["Stream", score_col], ascending=[True, False])
                                tmp["StreamRank"] = tmp.groupby("Stream").cumcount() + 1
                                st.session_state["ns_stream_rank_df"] = tmp
                except Exception:
                    pass
                # Optional: Trigger Map boost (soft weighting) for the fixed 39-play list
                apply_trigger_map = st.session_state.get("_apply_trigger_map", False)
                trigger_boost_points = float(st.session_state.get("_trigger_boost_points", 2.0) or 2.0)
                if apply_trigger_map and df_24 is not None and not df_24.empty and "BucketPick" in nl_df.columns:
                    try:
                        df_prev = df_24.copy()
                        # Use the last row per Stream as "previous winner" for that stream
                        if "Date" in df_prev.columns:
                            df_prev["_DateSort"] = _parse_date_series(df_prev["Date"])
                            df_prev = df_prev.sort_values(["Stream", "_DateSort"])
                        else:
                            df_prev = df_prev.sort_values(["Stream"])
                        prev_map = df_prev.groupby("Stream")["Result"].last().to_dict() if "Result" in df_prev.columns else {}
                        nl_df["PrevResult"] = nl_df["Stream"].map(prev_map).fillna("")
                        nl_df["TriggerBoost"] = nl_df.apply(
                            lambda r: trigger_map_boost(str(_safe_get(r,"BucketPick","")), str(_safe_get(r,"PrevResult","")), boost_points=trigger_boost_points),
                            axis=1,
                        )
                        nl_df["UniversalScore"] = nl_df["UniversalScore"].astype(float) + nl_df["TriggerBoost"].astype(float)
                    except Exception:
                        # Never break the playlist if trigger map cannot apply
                        pass
    
                # Ensure DuePressure exists (legacy compatibility)
                if "DuePressure" not in nl_df.columns:
                    if "DueBucketPressure" in nl_df.columns:
                        nl_df["DuePressure"] = nl_df["DueBucketPressure"].astype(float)
                    else:
                        nl_df["DuePressure"] = 0.0
    
                nl_df = nl_df.sort_values(["UniversalScore", "HitsPerWeek", "DuePressure"], ascending=[False, False, False]).reset_index(drop=True)
                nl_df.insert(0, "Rank", nl_df.index + 1)
                st.session_state["nl_df_current"] = nl_df.copy()
    
                st.dataframe(nl_df, width="stretch", height=520)
    
                # Northern Star percentile map (playlist)
                # This summarizes how much "hit weight" concentrates by rank position in the *final* per-stream playlist.
                with st.expander("Northern Star percentile map (playlist positions)", expanded=False):
                    try:
                        # Ensure DuePressure exists (legacy compatibility)
                        if "DuePressure" not in nl_df.columns:
                            if "DueBucketPressure" in nl_df.columns:
                                nl_df["DuePressure"] = nl_df["DueBucketPressure"].astype(float)
                            else:
                                nl_df["DuePressure"] = 0.0
                        # Keep only the best row per Stream (highest UniversalScore) -> one entry per stream
                        _best = nl_df.sort_values(
                            ["UniversalScore", "HitsPerWeek", "DuePressure"],
                            ascending=[False, False, False]
                        ).groupby("Stream", as_index=False).head(1).reset_index(drop=True)
    
                        # Assign playlist rank positions 1..N (typically 78 streams)
                        _best.insert(0, "RankPos", _best.index + 1)
                        _best["HitsWindow"] = _best.get("Hits", 0).astype(int)
    
                        _pos, _ = position_percentile_map(_best[["RankPos", "HitsWindow"]].copy())
                        st.caption("RankPos = position in the final per-stream playlist. HitCount = historical hits (in the selected window) of the #1 pick for that stream.")
                        st.dataframe(_pos, width="stretch", height=320)
                    except Exception as _e:
                        st.warning(f"Could not build the playlist percentile map: {_e}")
    
                # Northern Star buckets (per core)
                cfg = st.session_state.get("_cfg", RankConfig())
                with st.expander("Northern Star buckets (per core)", expanded=True):
                    if cores_for_cache:
                        _b_tabs = st.tabs([f"Core {c}" for c in cores_for_cache]) if len(cores_for_cache) > 1 else [st.container()]
                        for _tab, _c in zip(_b_tabs, cores_for_cache):
                            with _tab:
                                _core_str = str(_c).zfill(3)
                                _stats_df = compute_stream_stats(df_all, _core_str, window_days, exclude_md)
                                if _stats_df is None or _stats_df.empty:
                                    st.info(f"No AABC stream stats for core {_core_str}.")
                                    continue
                                _b = bucket_recommendations(_stats_df, cfg)
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.caption("Top 12 (BaseScore)")
                                    st.write(_b.get("Top12", []))
                                with c2:
                                    st.caption(f"Due {getattr(cfg, 'top_due', 8)} (DueIndex)")
                                    st.write(_b.get("Due8", []))
                                with c3:
                                    st.caption("Combined (Top+Due)")
                                    st.write(_b.get("Combined", []))
                    else:
                        st.info("Select one or more cores above to view buckets.")
    
                
                # Percentile map(s) for selected core(s) (tie-breaker visibility in Northern Lights view)
                with st.expander("Core ranking percentile map (tie-breaker)"):
                    if cores_for_cache:
                        _tabs = st.tabs([f"Core {c}" for c in cores_for_cache]) if len(cores_for_cache) > 1 else [st.container()]
                        for _tab, _c in zip(_tabs, cores_for_cache):
                            with _tab:
                                _core_str = str(_c).zfill(3)
                                # get_position_percentiles_cached() expects the active window + per-core stream stats.
                                # In this view we build / reuse the same stream-stats used by the core ranking.
                                _stream_stats = compute_stream_stats(df_all, _core_str, window_days, exclude_md)
                                _pm = get_position_percentiles_cached(_core_str, window_days, _stream_stats)
                                st.dataframe(_pm, width="stretch", height=240)
                    else:
                        st.info("Select one or more cores above to view percentile maps.")
    
    
                # Optional: straights shortlist (keep existing feature)
                if st.session_state.get("do_straights", False):
                    st.divider()
                    st.subheader("Generate straights shortlist (optional last)")
                    st.caption("This feature is unchanged; it runs only after the master playlist is built.")
                    try:
                        render_straights_shortlist(nl_df)
                    except Exception as e:
                        st.error(f"Straights shortlist failed: {e}")
    
    
    # --- Core view (single core or tabbed multi-core) ---
    
    if _t_ns is None:
        _t_ns = st.container()
    with _t_ns:
        st.header("Northern Star (v51)")
        st.caption("This tab restores the Northern Star scoring view and engines (Rare / Ultra-Rare) while keeping Core View unchanged. Percentile maps by position are shown here as GLOBAL (all selected cores) and PER-CORE maps.")
    
        # Global RankPos percentile map across selected cores (cache-only for safety)
        st.subheader("Global RankPos percentile map (all selected cores, cache-only)")
        ns_use_all_tracked = st.checkbox("Use ALL tracked cores (cache-only)", value=False, key="ns_use_all_tracked")
        ns_cores = CORE_PRESETS if ns_use_all_tracked else list(st.session_state.get("cores_for_cache_ms", []))
        if not ns_cores:
            st.info("Select cores in the multi-core section above (or enable ALL tracked cores).")
        else:
            expected_last = last_all if isinstance(last_all, str) else None
            global_map, missing = build_allcores_rankpos_pctmap(ns_cores, window_days=cfg.window_days, expected_last_date=expected_last, cache_only=True)
            if missing:
                st.error("Missing baseline caches for these cores (global map is cache-only): " + ", ".join(missing))
                st.caption("Build caches in the Cache Builder section, then rerun.")
            else:
                st.dataframe(global_map, width="stretch")
    
        st.divider()
        st.subheader("Per-core Northern Star scoring (SeedTraits + Cadence, soft)")
        view_core_ns = st.selectbox("Core (Northern Star view)", options=list(dict.fromkeys([view_core] + list(ns_cores))), key="view_core_ns")
        core_key_ns = canonical_core_key(view_core_ns)
    
        try:
            stats_ns = compute_stream_stats(df_all, core_key_ns, window_days=cfg.window_days, exclude_md=False)
        except Exception as e:
            st.error(f"Could not compute stats for core {core_key_ns}: {e}")
            stats_ns = pd.DataFrame()
    
        if stats_ns is not None and not stats_ns.empty:
            # Position map per-core (distinct from global map)
            pos_map_ns = get_position_percentiles_cached(core_key_ns, cfg.window_days, stats_ns)
            pos_strength_by_rank = dict(zip(pos_map_ns["RankPos"], pos_map_ns["PctStrength"]))
    
            # Cadence base: average gap for this core across streams (in days)
            total_hits = float(stats_ns["HitsWindow"].sum()) if "HitsWindow" in stats_ns.columns else 0.0
            mean_gap_days = (cfg.window_days / total_hits) if total_hits > 0 else 0.0
    
            # Seed Traits + Cadence per stream
            ns_rows = []
            for _, r in stats_ns.iterrows():
                stream = str(_safe_get(r,"Stream", ""))
                rankpos = int(_safe_get(r,"RankPos", 9999))
                pos_strength = float(pos_strength_by_rank.get(rankpos, 0.0))
                seed = _prev_seed_by_stream.get(stream)
                seed_score, seed_matches = compute_seed_traits_score(
                    core_key_ns, seed, stream,
                    pos_lookup=seed_traits_pos_lookup,
                    neg_lookup=seed_traits_neg_lookup,
                    last5_union_digits_by_stream=_last5_union_by_stream,
                )
                cadence = compute_cadence_score(float(_safe_get(r,"DaysSinceLastHit", 0.0)), mean_gap_days) if mean_gap_days > 0 else 0.0
    
                # Soft combined score
                hits_pw = float(_safe_get(r,"HitsPerWeek", 0.0))
                due_pressure = float(_safe_get(r,"DaysSinceLastHit", 0.0))
                ns_score = (
                    hits_pw
                    + (min(due_pressure, 50.0) * 0.01 * float(st.session_state.get("due_weight", 0.20)))
                    + (pos_strength * 0.01 * float(st.session_state.get("pos_weight", 0.25)))
                    + (seed_score * float(st.session_state.get("seed_traits_weight", 0.35)) if st.session_state.get("enable_seed_traits", True) else 0.0)
                    + (cadence * float(st.session_state.get("cadence_weight", 0.25)) if st.session_state.get("enable_cadence", True) else 0.0)
                )
                ns_rows.append({
                    "Stream": stream,
                    "RankPos": rankpos,
                    "HitsPerWeek": hits_pw,
                    "DaysSinceLastHit": due_pressure,
                    "PosPctStrength": pos_strength,
                    "SeedTraitsScore": seed_score,
                    "CadenceScore": cadence,
                    "NSScore": ns_score,
                    "Seed": seed,
                })
            ns_df = pd.DataFrame(ns_rows).sort_values(["NSScore","HitsPerWeek"], ascending=False)
    
            st.dataframe(ns_df.head(50), width="stretch")
            st.caption("NSScore is a soft additive score; it does NOT remove streams. Use it to prioritize without harming coverage.")
    
            with st.expander("Per-core RankPos percentile map (position-based)"):
                st.dataframe(pos_map_ns, width="stretch")
    
            # Engines (restored UI)
            st.divider()
            st.subheader("Rare Engine (AABC-family; historical lift)")
            if st.session_state.get("r1", True) or st.session_state.get("r2", True) or st.session_state.get("r3", True) or st.session_state.get("r4", True):
                # evaluate_rare_engine signature expects:
                #   (df_all, core, df_24h, enable_r1, enable_r2, enable_r3, enable_r4, window_days_recent)
                # Keep the UI-driven switches and pass the optional 24h map (may be empty).
                try:
                    try:
                        rare_df, _rare_summary = evaluate_rare_engine(
                        df_all,
                        core_key_ns,
                        df_24h,
                        enable_r1=r1,
                        enable_r2=r2,
                        enable_r3=r3,
                        enable_r4=r4,
                        window_days_recent=cfg.window_days,
                        )
                    except TypeError as _te:
                        if 'window_days_recent' in str(_te) and 'unexpected keyword argument' in str(_te):
                            rare_df, _rare_summary = evaluate_rare_engine(
                            df_all,
                            core_key_ns,
                            df_24h,
                            enable_r1=r1,
                            enable_r2=r2,
                            enable_r3=r3,
                            enable_r4=r4,
                            )
                        else:
                            raise
                    st.dataframe(_to_dataframe(rare_df), width="stretch")
                except Exception as e:
                    st.error(f"Rare Engine error: {e}")
                    st.dataframe(pd.DataFrame(), width="stretch")
            else:
                st.info("Enable at least one Rare Engine checkbox above to view results.")
    
            st.subheader("Ultra-Rare Engine (AABB/AAAB/etc; historical lift)")
            if st.session_state.get("q1", True) or st.session_state.get("q2", True) or st.session_state.get("q3", True) or st.session_state.get("q4", True):
                try:
                    try:
                        ultra_df, _ultra_summary = evaluate_ultra_rare_engine(
                        df_all,
                        core_key_ns,
                        df_24h,
                        enable_q1=q1,
                        enable_q2=q2,
                        enable_q3=q3,
                        enable_q4=q4,
                        window_days_recent=cfg.window_days,
                        )
                    except TypeError as _te:
                        if 'window_days_recent' in str(_te) and 'unexpected keyword argument' in str(_te):
                            ultra_df, _ultra_summary = evaluate_ultra_rare_engine(
                            df_all,
                            core_key_ns,
                            df_24h,
                            enable_q1=q1,
                            enable_q2=q2,
                            enable_q3=q3,
                            enable_q4=q4,
                            )
                        else:
                            raise
                    st.dataframe(_to_dataframe(ultra_df), width="stretch")
                except Exception as e:
                    st.error(f"Ultra-Rare Engine error: {e}")
                    st.dataframe(pd.DataFrame(), width="stretch")
            else:
                st.info("Enable at least one Ultra-Rare checkbox above to view results.")
    
            # Seed Traits match details (debug / transparency)
            with st.expander("Seed Traits matches (debug)"):
                pick_stream = st.selectbox("Stream to inspect", options=list(ns_df["Stream"].head(25)), key="ns_inspect_stream")
                seed = _prev_seed_by_stream.get(str(pick_stream))
                score, matches = compute_seed_traits_score(
                    core_key_ns, seed, str(pick_stream),
                    pos_lookup=seed_traits_pos_lookup,
                    neg_lookup=seed_traits_neg_lookup,
                    last5_union_digits_by_stream=_last5_union_by_stream,
                )
                st.write({"core": core_key_ns, "stream": str(pick_stream), "seed": seed, "score": score})
                if matches:
                    st.dataframe(pd.DataFrame(matches, columns=["trait","value","lift","sign"]), width="stretch")
                else:
                    st.caption("No matching traits found (or trait files not loaded).")
        else:
            st.info("No stats available for this core. Build or load data/caches and rerun.")
    
    
    if _t_core is None:
        _t_core = st.container()
with _t_core:
    st.subheader("Core view")

    if df_all is None or df_all.empty:
        st.info("Upload your history file first.")
        st.stop()

    if not cores_for_cache:
        st.info("Select one or more cores above to view core stats.")
        st.stop()

    show_tabs = st.checkbox(
        "Show tabs for all selected cores (optional)",
        value=False,
        key="show_tabs_for_all_selected_cores",
        help="If ON, you'll get a separate Core tab for each selected core. If OFF, you only see the core chosen in 'View core'.",
    )

    cfg = st.session_state.get("_cfg", RankConfig())

    def _render_one_core(core_id: str):
        core_id = str(core_id).zfill(3)
        st.markdown(f"### Core {core_id}")

        # Compute the AABC stream stats
        stats_df = compute_stream_stats(df_all, core_id, window_days=window_days, exclude_md=False)
        if stats_df is None or stats_df.empty:
            st.warning(f"No AABC stream stats found for core {core_id}.")
            return

        stats_df = stats_df.copy()
        st.subheader("Stream ranking (AABC doubles)")
        st.dataframe(stats_df, width="stretch", height=420)

        # Buckets (Top 12 BaseScore + Due 8 from ranks 13–60)
        # NOTE: build_northern_star_buckets() is a *per-stream* helper used by the master playlist.
        # For the per-core view we want the actual bucket lists, which are produced by bucket_recommendations().
        buckets = bucket_recommendations(stats_df, cfg)
        top_bucket = buckets.get("Top12", [])
        due_bucket = buckets.get("Due8", [])
        combined_bucket = buckets.get("Combined", [])

        st.subheader("Northern Star buckets")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Top 12 (BaseScore)")
            st.write(top_bucket)
        with c2:
            st.caption(f"Due {getattr(cfg, 'top_due', 8)} (DueIndex)")
            st.write(due_bucket)
        with c3:
            st.caption("Combined (Top+Due)")
            st.write(combined_bucket)

        # Core percentile map expander (tie-breaker)
        with st.expander("Core ranking percentile map (tie-breaker)", expanded=False):
            try:
                pos_map = get_position_percentiles_cached(core_id, window_days, stats_df)
                if pos_map is None or pos_map.empty:
                    st.info("No percentile map available for this core.")
                else:
                    st.dataframe(pos_map, width="stretch", height=420)
                    st.caption("Tip: use PctStrength as a soft tie-breaker when streams are close.")
            except Exception as e:
                st.error(f"Could not compute percentile map: {e}")

    if show_tabs:
        # Always render *all* selected cores in their own tabs
        core_tabs = st.tabs([str(c).zfill(3) for c in cores_for_cache])
        for c, t in zip(cores_for_cache, core_tabs):
            with t:
                _render_one_core(str(c))
    else:
        # Render only the currently selected view core
        _render_one_core(str(core_for_view))


# --- Backtest (optional) ---

# --- 7-Core Execution (Play Today): ranked by (stream, core) likelihood + $2 straight cap optimizer ---
if "_t_exec" in locals() and _t_exec is not None:
    with _t_exec:
        st.subheader("7-Core Execution (Play Today)")
        st.caption("Execution-focused for your 7 cores. Ranks **(stream, core)** lines by likelihood, applies your member rules, then selects the best **8** straight bets/day under a **$2.00 cap**.")

        exec_cores = ["246", "168", "589", "019", "468", "236", "025"]
        exec_cores_set = set(exec_cores)

        st.markdown("### Settings")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            straight_cap_dollars = st.number_input("Daily straight cap ($)", min_value=0.0, max_value=25.0, value=2.00, step=0.25, format="%.2f", key="exec_straight_cap")
        with c2:
            topk_per_member = st.selectbox("Candidate straights per played member (Top-K)", options=[1, 2, 3, 4, 6], index=2, key="exec_topk_per_member")
        with c3:
            min_topk_coverage = st.slider("Min Top-K coverage to allow straights (training slice)", min_value=0.10, max_value=0.80, value=0.30, step=0.01, key="exec_min_cov")

        # Cadence boost (least-risk): uses uploaded cadence table only to slightly reorder ranked lines in this tab.
        use_cadence_boost = st.checkbox(
            "Use cadence boost in 7-core ranking (from cadence upload)",
            value=True,
            key="exec_use_cadence_boost",
            help="If enabled and a cadence TABLE is uploaded, this tab will apply a small 'due' boost using DaysSinceLastHit vs typical gap. This does not change the Northern Star engine; only this Execution tab."
        )
        cadence_boost_weight = st.slider(
            "Cadence boost weight (small additive)",
            min_value=0.00,
            max_value=0.50,
            value=0.15,
            step=0.01,
            key="exec_cadence_boost_weight",
            help="Caps how much cadence can move the ranking. Keep this small to avoid regressions."
        )

        max_straight_bets = int((straight_cap_dollars // 0.25) if straight_cap_dollars else 0)
        st.info(f"Straight cap = **${straight_cap_dollars:.2f}/day** ⇒ max straight bets/day = **{max_straight_bets}** at $0.25 each.")

        play_date = None
        last_dt = None
        try:
            if "Date" in df_all.columns and not df_all.empty:
                last_dt = pd.to_datetime(df_all["Date"]).max()
                play_date = (last_dt + pd.Timedelta(days=1)).date()
                st.write(f"**Play date:** {play_date} (day after last history date: {last_dt.date()})")
        except Exception:
            pass

        st.markdown("### Ranked (Stream, Core) lines")
        st.caption("This uses the app’s existing Northern Star stream table if available (no changes to your ranking engine).")

        ranked_lines = None
        try:
            def _normalize_ns_rank(df):
                if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
                    return None
                out = df.copy()
                # normalize columns (older cache builds used 'Core' not 'core')
                if "core" not in out.columns and "Core" in out.columns:
                    out["core"] = out["Core"]
                if "Stream" not in out.columns and "stream" in out.columns:
                    out["Stream"] = out["stream"]
                if "core" in out.columns:
                    out["core"] = out["core"].astype(str).str.zfill(3)
                return out

            cand = _normalize_ns_rank(st.session_state.get("ns_stream_rank_df"))

            # window-specific in-memory fallback (set by cache builder)
            if cand is None:
                cand = _normalize_ns_rank(st.session_state.get(f"ns_stream_rank_df_{window_days}"))

            # disk fallback (survives reruns / tab switches on Streamlit Cloud)
            if cand is None:
                cand = _normalize_ns_rank(_load_ns_stream_rank_from_disk(int(window_days)))

            # try alternate windows (common: 180/365)
            if cand is None:
                for _w in [180, 365]:
                    if int(_w) == int(window_days):
                        continue
                    cand = _normalize_ns_rank(st.session_state.get(f"ns_stream_rank_df_{_w}"))
                    if cand is None:
                        cand = _normalize_ns_rank(_load_ns_stream_rank_from_disk(int(_w)))
                    if cand is not None:
                        break

            if cand is not None:
                st.session_state["ns_stream_rank_df"] = cand
                try:
                    st.session_state[f"ns_stream_rank_df_{window_days}"] = cand
                except Exception:
                    pass

                tmp = cand.copy()
                if "core" in tmp.columns:
                    tmp["core"] = tmp["core"].astype(str).str.zfill(3)
                    tmp = tmp[tmp["core"].isin(exec_cores)]

                if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                    score_col = None
                    for col in ["BaseScore", "Score", "score", "basescore"]:
                        if col in tmp.columns:
                            score_col = col
                            break
                    if score_col is None:
                        if "RankPos" in tmp.columns:
                            tmp["__SCORE__"] = 1.0 / (tmp["RankPos"].astype(float).clip(lower=1.0))
                            score_col = "__SCORE__"
                        else:
                            tmp["__SCORE__"] = 0.0
                            score_col = "__SCORE__"
                    ranked_lines = tmp.sort_values(by=[score_col], ascending=False).reset_index(drop=True)
        except Exception:
            ranked_lines = None


        # --- Enrich ranked_lines for breakdown/debug (NO changes to ranking engine) ---
        if ranked_lines is not None and len(ranked_lines) > 0:
            # 1) Robust cadence merge (stream-key canonicalization)
            if cadence_table_df is not None and not cadence_table_df.empty:
                try:
                    def _canon_stream(s: str) -> str:
                        s = "" if s is None else str(s)
                        s = s.replace("\u00a0", " ")
                        s = re.sub(r"\s*\|\s*", " | ", s.strip())
                        s = re.sub(r"\s+", " ", s)
                        return s.lower().strip()

                    rl = ranked_lines.copy()
                    rl["__stream_key"] = rl["Stream"].astype(str).map(_canon_stream)
                    rl["__core_key"] = rl["core"].astype(str).str.strip().map(lambda x: x.zfill(3) if str(x).isdigit() else str(x))

                    cad = cadence_table_df.copy()
                    # cadence_table_df is already normalized by _parse_cadence_table_from_text()
                    if "stream_key" not in cad.columns:
                        cad["stream_key"] = cad["stream"].astype(str).map(_canon_stream)
                    cad["__stream_key"] = cad["stream_key"]

                    if "core_family" in cad.columns:
                        cad["__core_key"] = cad["core_family"].astype(str).str.strip().map(lambda x: x.zfill(3) if str(x).isdigit() else str(x))
                    else:
                        cad["__core_key"] = "*"

                    # Prefer core-specific cadence rows if present; else stream-only rows apply to all cores
                    cad_core = cad[cad["__core_key"] != "*"].copy()
                    cad_stream = cad[cad["__core_key"] == "*"].copy()
                    # If uploaded cadence file doesn't include a precomputed CadenceScore (common),
                    # derive it deterministically from the cadence file's real stats.
                    def _ensure_cadence_cols(df: pd.DataFrame) -> pd.DataFrame:
                        out = df.copy()
                        # Map cadence export column names into the canonical ones used by this app.
                        cols_lower = {c.lower(): c for c in out.columns}
                        def _col(*names):
                            for n in names:
                                if n in out.columns:
                                    return n
                                if n.lower() in cols_lower:
                                    return cols_lower[n.lower()]
                            return None
                        
                        stream_col = _col('Stream', 'stream', 'stream_name', 'streamname')
                        core_col = _col('Core', 'core', 'core_family', 'corefamily')
                        member_col = _col('Member', 'member')
                        dslh_col = _col('DaysSinceLastHit', 'dayssincelasthit', 'days_since_last_hit', 'dslh')
                        mg_col = _col('MeanGapDays', 'mean_gap_days', 'meangapdays', 'mean_gap', 'avg_gap_days')
                        hpw_col = _col('HitsPerWeek', 'hitsperweek', 'hits_per_week')
                        
                        if stream_col and 'Stream' not in out.columns:
                            out['Stream'] = out[stream_col]
                        if core_col and 'Core' not in out.columns:
                            out['Core'] = out[core_col]
                        if member_col and 'Member' not in out.columns:
                            out['Member'] = out[member_col]
                        if dslh_col and 'DaysSinceLastHit' not in out.columns:
                            out['DaysSinceLastHit'] = out[dslh_col]
                        
                        # Ensure both MeanGapDays and mean_gap_days exist if either is present.
                        if mg_col and ('MeanGapDays' not in out.columns and 'mean_gap_days' not in out.columns):
                            out['MeanGapDays'] = out[mg_col]
                            out['mean_gap_days'] = out[mg_col]
                        elif 'MeanGapDays' in out.columns and 'mean_gap_days' not in out.columns:
                            out['mean_gap_days'] = out['MeanGapDays']
                        elif 'mean_gap_days' in out.columns and 'MeanGapDays' not in out.columns:
                            out['MeanGapDays'] = out['mean_gap_days']
                        
                        if hpw_col and 'HitsPerWeek' not in out.columns:
                            out['HitsPerWeek'] = out[hpw_col]
                        
                        if 'Stream' not in out.columns:
                            out['Stream'] = ''
                        if 'Core' not in out.columns:
                            out['Core'] = ''
                        out['Core'] = out['Core'].astype(str).str.strip().str.zfill(3)
                        
                        # Canonical stream key used for merges with ranked_lines.
                        out['stream_key'] = out['Stream'].astype(str).apply(_canon_stream)
                        if '__stream_key' not in out.columns:
                            out['__stream_key'] = out['stream_key']
                        if '__core_key' not in out.columns:
                            out['__core_key'] = out['Core']
                        
                        # Build a reasonable Hits180 proxy if missing.
                        if 'Hits180' not in out.columns:
                            if 'HitsPerWeek' in out.columns:
                                out['Hits180'] = (pd.to_numeric(out['HitsPerWeek'], errors='coerce') * (180.0/7.0)).fillna(0.0)
                            else:
                                out['Hits180'] = 0.0
                        
                        # Compute CadenceScore if absent (this is what feeds CadTerm in the 7-core breakdown).
                        if ('CadenceScore' not in out.columns) or out['CadenceScore'].isna().all():
                            if 'DaysSinceLastHit' in out.columns and 'MeanGapDays' in out.columns:
                                out['CadenceScore'] = out.apply(
                                    lambda r: compute_cadence_score(_safe_get(r,'DaysSinceLastHit'), _safe_get(r,'MeanGapDays')),
                                    axis=1
                                )
                            else:
                                dslh = pd.to_numeric(out.get('DaysSinceLastHit', pd.Series([np.nan]*len(out))), errors='coerce')
                                hits180 = pd.to_numeric(out.get('Hits180', pd.Series([0]*len(out))), errors='coerce').fillna(0.0)
                                out['CadenceScore'] = ((hits180 > 0) & (dslh <= 180)).astype(float)
                        
                        return out
                    cad_core = _ensure_cadence_cols(cad_core)
                    cad_stream = _ensure_cadence_cols(cad_stream)


                    merged = rl
                    if not cad_core.empty:
                        merged = merged.merge(
                            cad_core[["__stream_key","__core_key","DaysSinceLastHit","mean_gap_days","CadenceScore"]],
                            how="left",
                            on=["__stream_key","__core_key"],
                            suffixes=("","_cad")
                        )
                    if not cad_stream.empty:
                        merged = merged.merge(
                            cad_stream[["__stream_key","DaysSinceLastHit","mean_gap_days","CadenceScore"]].rename(
                                columns={
                                    "DaysSinceLastHit":"DaysSinceLastHit_stream",
                                    "mean_gap_days":"mean_gap_days_stream",
                                    "CadenceScore":"CadenceScore_stream",
                                }
                            ),
                            how="left",
                            on="__stream_key",
                        )
                        # fill from stream-only where core-specific missing
                        merged["DaysSinceLastHit"] = merged["DaysSinceLastHit"].fillna(merged["DaysSinceLastHit_stream"])
                        merged["mean_gap_days"] = merged["mean_gap_days"].fillna(merged["mean_gap_days_stream"])
                        merged["CadenceScore"] = merged["CadenceScore"].fillna(merged["CadenceScore_stream"])

                        merged = merged.drop(columns=[c for c in ["DaysSinceLastHit_stream","mean_gap_days_stream","CadenceScore_stream"] if c in merged.columns])

                    # If cadence score could not be computed during parse, compute now
                    if "CadenceScore" not in merged.columns:
                        merged["CadenceScore"] = 0.0
                    merged["CadenceScore"] = pd.to_numeric(merged["CadenceScore"], errors="coerce").fillna(0.0)

                    ranked_lines = merged.drop(columns=[c for c in ["__stream_key","__core_key"] if c in merged.columns])
                except Exception:
                    # Keep app running; cadence terms remain 0 if merge fails
                    if "CadenceScore" not in ranked_lines.columns:
                        ranked_lines["CadenceScore"] = 0.0

            else:
                if "CadenceScore" not in ranked_lines.columns:
                    ranked_lines["CadenceScore"] = 0.0

            # 2) Ensure PctStrength exists (strict percentile of BaseScore within each core)
            if "PctStrength" not in ranked_lines.columns or pd.to_numeric(ranked_lines.get("PctStrength", 0), errors="coerce").fillna(0).eq(0).all():
                if "PosPctStrength" in ranked_lines.columns:
                    ranked_lines["PctStrength"] = pd.to_numeric(ranked_lines["PosPctStrength"], errors="coerce").fillna(0.0)
                else:
                    # Strict percentile: 100 * (# strictly lower BaseScore) / N
                    if "BaseScore" in ranked_lines.columns:
                        base_col = "BaseScore"
                    else:
                        base_col = "HitsPerWeek" if "HitsPerWeek" in ranked_lines.columns else None

                    if base_col is not None:
                        def _strict_pct(g: pd.DataFrame) -> pd.Series:
                            x = pd.to_numeric(g[base_col], errors="coerce").fillna(0.0)
                            n = max(len(x), 1)
                            r = x.rank(method="min", ascending=True)  # 1..n
                            return (r - 1) / n * 100.0
                        ranked_lines["PctStrength"] = ranked_lines.groupby("core", dropna=False, sort=False).apply(
                            lambda g: _strict_pct(g)
                        ).reset_index(level=0, drop=True).astype(float)
                    else:
                        ranked_lines["PctStrength"] = 0.0
            ranked_lines["PctStrength"] = pd.to_numeric(ranked_lines["PctStrength"], errors="coerce").fillna(0.0)

            # 3) Guarantee debug score columns exist (do NOT alter ordering/ranking output)
            for col in ["SeedTraitsScore","UniversalScore"]:
                if col not in ranked_lines.columns:
                    ranked_lines[col] = 0.0
                ranked_lines[col] = pd.to_numeric(ranked_lines[col], errors="coerce").fillna(0.0)

            # 4) Recompute the terms for the debug table (UniversalScore & RecomputedScore shown together)
            # Weights are already defined above in this tab.
            base = pd.to_numeric(ranked_lines.get("BaseScore", ranked_lines.get("HitsPerWeek", 0.0)), errors="coerce").fillna(0.0)
            due  = pd.to_numeric(ranked_lines.get("DueIndex", ranked_lines.get("DueScore", 0.0)), errors="coerce").fillna(0.0)
            pos  = ranked_lines["PctStrength"] / 100.0

            # --- Weights (must be defined before recomputing terms in this tab) ---
            # BaseTerm uses fixed base_w=1.0; other weights come from sidebar sliders (session_state keys).
            base_w = 1.0
            due_w = float(st.session_state.get("due_weight", 1.0))
            pos_w = float(st.session_state.get("pos_weight", 1.0))
            seed_traits_weight = float(st.session_state.get("seed_traits_weight", 0.0))
            cadence_weight = float(st.session_state.get("cadence_weight", 0.0))

            # ---------- SeedTraits enrich (7-core execution path) ----------
            # Ensure we have a Seed per Stream so we can score seed traits at the (stream, core) line level.
            if "Seed" not in ranked_lines.columns:
                try:
                    ranked_lines["Seed"] = ranked_lines[stream_col].map(_prev_seed_by_stream)
                except Exception:
                    ranked_lines["Seed"] = ""
            ranked_lines["Seed"] = (
                ranked_lines["Seed"]
                .fillna("")
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.strip()
                .str.zfill(4)
            )

            # Compute SeedTraitsScore (soft additive)
            # IMPORTANT: lookups may live in globals() OR st.session_state depending on upload path.
            try:
                _enable_seed_traits_boost = bool(enable_seed_traits_boost)
            except Exception:
                _enable_seed_traits_boost = bool(st.session_state.get("enable_seed_traits_boost", False))

            _pos_lookup = globals().get("seed_traits_pos_lookup", None)
            _neg_lookup = globals().get("seed_traits_neg_lookup", None)
            if _pos_lookup is None:
                _pos_lookup = st.session_state.get("seed_traits_pos_lookup", None)
            if _neg_lookup is None:
                _neg_lookup = st.session_state.get("seed_traits_neg_lookup", None)

            if "SeedTraitsScore" not in ranked_lines.columns:
                ranked_lines["SeedTraitsScore"] = 0.0

            if _enable_seed_traits_boost and (_pos_lookup is not None) and (_neg_lookup is not None):
                # Need Seed column to score; if missing, we cannot apply seed traits.
                if "Seed" in ranked_lines.columns and ("Stream" in ranked_lines.columns):
                    _core_col = core_col if ("core_col" in locals() and core_col in ranked_lines.columns) else ("Core" if "Core" in ranked_lines.columns else None)
                    if _core_col is not None:
                        # compute raw (score, matches) tuples
                        _seedtraits_raw = ranked_lines.apply(
                            lambda r: compute_seed_traits_score(
                                seed=str(_safe_get(r,"Seed","")).zfill(4),
                                core=str(_safe_get(r,_core_col,"")).zfill(3),
                                stream=str(_safe_get(r,"Stream","")),
                                pos_lookup=_pos_lookup,
                                neg_lookup=_neg_lookup,
                                weight_pos=1.0,
                                weight_neg=1.0,
                            ),
                            axis=1,
                        )

                        # Store debug matches (if returned) WITHOUT breaking numeric scoring.
                        if "SeedTraitsMatches" not in ranked_lines.columns:
                            ranked_lines["SeedTraitsMatches"] = None

                        def _st_score(x):
                            if isinstance(x, dict):
                                return x.get("score", 0.0)
                            if isinstance(x, (tuple, list)):
                                return x[0] if len(x) > 0 else 0.0
                            return x

                        def _st_matches(x):
                            if isinstance(x, dict):
                                return x.get("matches", None)
                            if isinstance(x, (tuple, list)):
                                return x[1] if len(x) > 1 else None
                            return None

                        ranked_lines["SeedTraitsScore"] = pd.to_numeric(_seedtraits_raw.apply(_st_score), errors="coerce").fillna(0.0)
                        ranked_lines["SeedTraitsMatches"] = _seedtraits_raw.apply(_st_matches)

                # else: keep zeros (no Seed column)

            # -----------------------------------------------------------------
            # Ensure the per-term columns exist for the Ranked (Stream, Core) table.
            # These are DISPLAY/DEBUG terms only; they mirror the NSScore formula used
            # for ranking in this view and must never crash the app.
            # -----------------------------------------------------------------
            hits_pw = pd.to_numeric(ranked_lines.get("HitsPerWeek", 0.0), errors="coerce").fillna(0.0)
            dslh = pd.to_numeric(ranked_lines.get("DaysSinceLastHit", 0.0), errors="coerce").fillna(0.0)
            pos_pct = pd.to_numeric(
                ranked_lines["PosPctStrength"] if "PosPctStrength" in ranked_lines.columns else ranked_lines.get("PctStrength", 0.0),
                errors="coerce",
            ).fillna(0.0)
            seed_score = pd.to_numeric(ranked_lines.get("SeedTraitsScore", 0.0), errors="coerce").fillna(0.0)
            cad_score = pd.to_numeric(ranked_lines.get("CadenceScore", 0.0), errors="coerce").fillna(0.0)

            base_w = float(st.session_state.get("base_w", 1.0))
            due_w  = float(st.session_state.get("due_w", 1.0))
            pos_w  = float(st.session_state.get("pos_w", 1.0))
            seed_w = float(st.session_state.get("seed_traits_weight", 0.35))
            cad_w  = float(st.session_state.get("cadence_weight", 0.25))
            use_seed = bool(st.session_state.get("enable_seed_traits_boost", st.session_state.get("enable_seed_traits", True)))
            use_cad  = bool(st.session_state.get("enable_cadence_boost", st.session_state.get("enable_cadence", True)))

            ranked_lines["BaseTerm"] = base_w * hits_pw
            ranked_lines["DueTerm"]  = (dslh.clip(upper=50.0) * 0.01) * due_w
            ranked_lines["PosTerm"]  = (pos_pct * 0.01) * pos_w
            ranked_lines["SeedTerm"] = (seed_score * seed_w) if use_seed else 0.0
            ranked_lines["CadTerm"]  = (cad_score * cad_w) if use_cad else 0.0

            if "UniversalScore" not in ranked_lines.columns:
                ranked_lines["UniversalScore"] = ranked_lines.get("NSScore", 0.0)
            # Columns to show (only those that exist)
            show_cols_pref = [
                "Stream","core",
                "UniversalScore","RecomputedScore",
                "BaseTerm","DueTerm","PosTerm","SeedTerm","CadTerm",
                "HitsPerWeek","DaysSinceLastHit","PctStrength",
                "SeedTraitsScore","CadenceScore",
            ]
            show_cols = [c for c in show_cols_pref if c in ranked_lines.columns]
            ranked_lines["RecomputedScore"] = (
                ranked_lines["BaseTerm"] + ranked_lines["DueTerm"] + ranked_lines["PosTerm"] +
                ranked_lines["SeedTerm"] + ranked_lines["CadTerm"]
            ).astype(float)

            # If UniversalScore is missing/zero but recomputed is available, show recomputed as the effective score
            if pd.to_numeric(ranked_lines["UniversalScore"], errors="coerce").fillna(0.0).eq(0).all():
                ranked_lines["UniversalScore"] = ranked_lines["RecomputedScore"].astype(float)

            # Keep DaysSinceLastHit column aligned with cadence-provided DSLH if present
            if "DaysSinceLastHit" in ranked_lines.columns:
                ranked_lines["DaysSinceLastHit"] = pd.to_numeric(ranked_lines["DaysSinceLastHit"], errors="coerce").fillna(0).astype(int)
            st.dataframe(ranked_lines[show_cols].head(40), use_container_width=True, hide_index=True)

            with st.expander("Score breakdown (7-core execution) — explain UniversalScore", expanded=False):
                st.caption("This is *read-only* debug: it does NOT change ranking. It decomposes the score pieces used to build `UniversalScore` for each (stream, core) line.")
                needed = {"UniversalScore","HitsPerWeek","DaysSinceLastHit","PctStrength","SeedTraitsScore","CadenceScore"}
                have = set(ranked_lines.columns)
                missing = sorted(list(needed - have))
                if missing:
                    st.info(f"Breakdown unavailable for these rows (missing columns: {', '.join(missing)}). Showing whatever score columns exist.")
                    dbg_cols = [c for c in ["UniversalScore","HitsPerWeek","DuePressure","PosPctStrength","SeedTraitsScore","CadenceScore","DaysSinceLastHit","PctStrength"] if c in ranked_lines.columns]
                    if dbg_cols:
                        st.dataframe(ranked_lines[dbg_cols].head(40), use_container_width=True, hide_index=True)
                else:
                    base_w = float(st.session_state.get("ns_base_weight", 1.0))
                    due_w  = float(st.session_state.get("ns_due_weight",  1.0))
                    pos_w  = float(st.session_state.get("ns_pos_weight",  1.0))
                    st_w   = float(st.session_state.get("seed_traits_weight", 0.25))
                    cad_w  = float(st.session_state.get("cadence_weight", 0.20))

                    use_seed = bool(st.session_state.get("enable_seed_traits_boost", True))
                    use_cad  = bool(st.session_state.get("enable_cadence_boost", True))

                    tmp = ranked_lines.copy()
                    tmp["BaseTerm"] = tmp["HitsPerWeek"].astype(float) * base_w
                    tmp["DueTerm"]  = (tmp["DaysSinceLastHit"].astype(float).clip(lower=0, upper=50) * 0.01) * due_w
                    tmp["PosTerm"]  = (tmp["PctStrength"].astype(float) * 0.01) * pos_w
                    tmp["SeedTerm"] = (tmp["SeedTraitsScore"].astype(float) * (st_w if use_seed else 0.0))
                    tmp["CadTerm"]  = (tmp["CadenceScore"].astype(float) * (cad_w if use_cad else 0.0))
                    tmp["RecomputedScore"] = tmp[["BaseTerm","DueTerm","PosTerm","SeedTerm","CadTerm"]].sum(axis=1)
                    cols = [
                        "Stream","core","UniversalScore","RecomputedScore",
                        "BaseTerm","DueTerm","PosTerm","SeedTerm","CadTerm",
                        "HitsPerWeek","DaysSinceLastHit","PctStrength","SeedTraitsScore","CadenceScore"
                    ]
                    cols = [c for c in cols if c in tmp.columns]
                    st.dataframe(tmp[cols].head(40), use_container_width=True, hide_index=True)

                    st.markdown("**Weights in effect for this run**")
                    st.write({
                        "base_w": base_w,
                        "due_w": due_w,
                        "pos_w": pos_w,
                        "seed_traits_weight": st_w if use_seed else 0.0,
                        "cadence_weight": cad_w if use_cad else 0.0,
                        "enable_seed_traits_boost": use_seed,
                        "enable_cadence_boost": use_cad,
                    })

            with st.expander("Traits audit (7-core execution) — prove traits are firing", expanded=False):
                st.caption("This panel does NOT change ranking. It shows which traits matched for the current stream seeds for the lines above (core-level seed traits).")
                try:
                    if ranked_lines is None or ranked_lines.empty:
                        st.write("No ranked lines available.")
                    else:
                        # Build a small audit sample from the same ranked_top used below (up to 25 lines)
                        _audit = ranked_lines.copy().head(25)
                        # Need seeds per stream
                        seed_map = globals().get("_prev_seed_by_stream", {}) if isinstance(globals().get("_prev_seed_by_stream", None), dict) else {}
                        last5_map = globals().get("_last5_union_by_stream", {}) if isinstance(globals().get("_last5_union_by_stream", None), dict) else {}
                        rows = []
                        for _, rr in _audit.iterrows():
                            s = str(rr.get("Stream","")).strip()
                            c = str(rr.get("core","")).zfill(3)
                            seed = seed_map.get(s)
                            score, matches = compute_seed_traits_score(
                                c,
                                seed,
                                s,
                                pos_lookup=core_traits_pos_lookup,
                                neg_lookup=core_traits_neg_lookup,
                                last5_union_digits_by_stream=last5_map,
                                cap=2.0,
                            )
                            npos = sum(1 for t,v,l,sgn in matches if sgn == "+")
                            nneg = sum(1 for t,v,l,sgn in matches if sgn == "-")
                            topm = "; ".join([f"{t}={v}({sgn}{l:.2f})" for t,v,l,sgn in matches[:6]])
                            rows.append({
                                "Stream": s,
                                "Core": c,
                                "SeedUsed": seed if seed is not None else "",
                                "TraitScore": round(float(score), 3),
                                "Matches_Pos": int(npos),
                                "Matches_Neg": int(nneg),
                                "TopMatches": topm,
                            })
                        audit_df = pd.DataFrame(rows)
                        st.dataframe(audit_df, use_container_width=True, hide_index=True)
                        # Quick coverage stats
                        try:
                            cov = float((audit_df["Matches_Pos"] + audit_df["Matches_Neg"] > 0).mean())
                            st.write(f"Trait match coverage in sample: **{cov*100:.1f}%** of lines have ≥1 matched trait.")
                        except Exception:
                            pass
                except Exception as e:
                    st.write("Traits audit failed to render:", e)

        else:
            # If the user built the cache but the stream-rank table isn't present (session reset / disk write unavailable),
            # rebuild it on-demand from the same cached per-core stream stats logic used by the cache builder.
            try:
                cores_for_rank = list(dict.fromkeys([int(c) for c in (selected_cores_7 or [])])) if "selected_cores_7" in locals() else []
            except Exception:
                cores_for_rank = []
            if not cores_for_rank:
                try:
                    cores_for_rank = list(dict.fromkeys([int(c) for c in cores_for_cache])) if "cores_for_cache" in locals() else []
                except Exception:
                    cores_for_rank = []
            try:
                if df_all is not None and not df_all.empty and cores_for_rank:
                    rows = []
                    for core in cores_for_rank:
                        sdf = get_stream_stats_cached(df_all, core, window_days, exclude_md=exclude_md)
                        if sdf is None or sdf.empty:
                            continue
                        sdf = sdf.copy()
                        sdf["core"] = int(core)
                        rows.append(sdf)
                    if rows:
                        tmp_rank = pd.concat(rows, ignore_index=True)
                        # Match the cache-builder scoring exactly (Base+Due+Pos) for the (stream, core) lines
                        score_col = "UniversalScore"
                        tmp_rank[score_col] = (
                            tmp_rank.get("BaseScore", 0).fillna(0).astype(float)
                            + tmp_rank.get("DueIndex", 0).fillna(0).astype(float)
                            + tmp_rank.get("PosPctStrength", 0).fillna(0).astype(float) / 100.0
                        )
                        cand = tmp_rank.sort_values(by=[score_col], ascending=False).reset_index(drop=True)
                        st.session_state[f"ns_stream_rank_df_{window_days}"] = cand
                        try:
                            _save_ns_stream_rank_to_disk(cand, window_days)
                        except Exception:
                            pass
            except Exception:
                cand = None

            if cand is None or len(cand) == 0:
                st.warning("No cached Northern Star stream ranking table found. Run Northern Star cache build first, then return here.")

        st.markdown("### Box plan + $2 straight optimizer")
        st.caption("Member rule: **246 + 168 = ALL 3 members**, others = **Top 2**. Straights: pick top-probability straights across all candidates until the daily cap is filled.")

        if ranked_lines is not None and not ranked_lines.empty and play_date is not None:
            hist_train = df_all.copy()
            hist_train["Date"] = pd.to_datetime(hist_train["Date"])
            hist_train = hist_train[hist_train["Date"].dt.date < play_date].copy()

            pick_rows = []
            straight_candidates = []  # (p, straight, stream, core, member_box, source, n, cov)

            topN = min(60, len(ranked_lines))
            ranked_top = ranked_lines.head(topN)

            for _, r in ranked_top.iterrows():
                stream = str(_safe_get(r,"Stream", "")).strip()
                core = str(_safe_get(r,"core", "")).zfill(3)
                if core not in exec_cores_set:
                    continue

                all3 = core in {"246", "168"}
                members = members_from_core(core, "AABC")  # [AABC, ABBC, ABCC] box-keys
                played = members if all3 else members[:2]

                pick_rows.append({
                    "Stream": stream,
                    "Core": core,
                    "MembersPlayed": ", ".join(played),
                    "All3": "YES" if all3 else "NO",
                })

                for mb in played:
                    items, meta = top_straights_for_box_from_history(
                        hist_train,
                        member_box4=mb,
                        stream_key=stream,
                        top_k=int(topk_per_member),
                        min_stream_samples=20,
                        min_global_samples=50,
                    )
                    if not items:
                        continue
                    cov = float(sum([p for (_, p, _, _) in items]))
                    if cov < float(min_topk_coverage):
                        continue
                    for straight, p, n, used in items:
                        straight_candidates.append((float(p), straight, stream, core, mb, used, int(n), cov))

            if pick_rows:
                st.markdown("#### Box plan (top-ranked candidates)")
                
                _box_df = pd.DataFrame(pick_rows)
                # If a cadence *table* was uploaded (CSV-formatted), merge stream/core cadence into the Play Today box plan
                try:
                    if isinstance(globals().get("cadence_table_df", None), pd.DataFrame) and (not globals()["cadence_table_df"].empty):
                        _cad = globals()["cadence_table_df"].copy()
                        _cad = _cad[_cad.get("level","").astype(str).str.lower().eq("core")] if "level" in _cad.columns else _cad
                        # Expect cadence table uses column 'stream' not 'Stream'
                        if "stream" in _cad.columns:
                            _cad = _cad.rename(columns={"stream":"Stream"})
                        _cad["Core"] = _cad["core_family"].astype(str).str.zfill(3)
                        # Pick useful cols if present
                        keep_cols = ["Stream","Core"]
                        for col in ["days_since_last_hit","median_gap_days","hits_last90","hits_last180","hits_last365"]:
                            if col in _cad.columns:
                                keep_cols.append(col)
                        _cad2 = _cad[keep_cols].drop_duplicates(subset=["Stream","Core"], keep="first")
                        _box_df["Core"] = _box_df["Core"].astype(str).str.zfill(3)
                        _box_df = _box_df.merge(_cad2, on=["Stream","Core"], how="left")
                except Exception:
                    pass

                st.dataframe(_box_df.head(40), use_container_width=True, hide_index=True)



            # Download Box plan (includes cadence cols if merged)
            try:
                if 'pick_rows' in locals() and pick_rows:
                    _dl_box = pd.DataFrame(pick_rows)
                    try:
                        if "_box_df" in locals() and isinstance(_box_df, pd.DataFrame) and not _box_df.empty:
                            _dl_box = _box_df.copy()
                    except Exception:
                        pass
                    st.download_button(
                        "Download box plan (CSV)",
                        data=_dl_box.to_csv(index=False).encode("utf-8"),
                        file_name=f"play_today_box_plan_{play_date}.csv" if play_date else "play_today_box_plan.csv",
                        mime="text/csv",
                        key="dl_play_today_box_plan",
                    )
            except Exception:
                pass

            st.markdown("#### Straights add-on (optimized under cap)")
            if max_straight_bets <= 0:
                st.info("Straight cap is $0.00 — no straights will be selected.")
            else:
                if not straight_candidates:
                    st.info("No eligible straight candidates passed sample-size + coverage gates.")
                else:
                    straight_candidates.sort(key=lambda x: (x[0], x[7], x[6]), reverse=True)

                    selected = []
                    seen = set()
                    for p, straight, stream, core, mb, used, n, cov in straight_candidates:
                        key = (straight, stream)
                        if key in seen:
                            continue
                        selected.append({
                            "Stream": stream,
                            "Core": core,
                            "MemberBox": mb,
                            "Straight": straight,
                            "p": round(p, 4),
                            "TopK_Coverage": round(cov, 4),
                            "SamplesUsed": n,
                            "Source": used,
                        })
                        seen.add(key)
                        if len(selected) >= max_straight_bets:
                            break

                    if selected:
                        st.dataframe(pd.DataFrame(selected), use_container_width=True, hide_index=True)
                        total_cost = 0.25 * len(selected)
                        st.success(f"Selected **{len(selected)}** straight(s). Straight cost = **${total_cost:.2f}** (cap ${straight_cap_dollars:.2f}).")
                    else:
                        st.info("No unique straight candidates could be selected under current gates.")


# --- Validator (walk-forward Top2) ---
if "_t_val" in locals():
    if _t_val is None:
        _t_val = st.container()
    with _t_val:
        try:
            render_validator_tab(df_all=df_all)
        except Exception as e:
            st.error(f"Validator failed: {e}")
            with st.expander("Show full traceback"):
                st.code(traceback.format_exc())



if _t_bt is None:
    _t_bt = st.container()
with _t_bt:
    st.subheader("Backtest (optional)")
    st.caption("Optional diagnostics. This does not change your core ranking output.")

    if df_all is None or df_all.empty:
        st.info("Upload your history file first.")
        st.stop()

    if not cores_for_cache:
        st.info("Select one or more cores above to backtest.")
        st.stop()

    try:
        render_backtest(df_all=df_all, cfg=cfg, cores_for_cache=cores_for_cache, df_24h=df_24h)
    except NameError as e:
        st.error(f"Backtest failed (NameError): {e}")
        with st.expander('Show full traceback'):
            st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        with st.expander('Show full traceback'):
            st.code(traceback.format_exc())
# Northern Star (core) RankPos map is the same distribution, but we cache it separately for clarity
_core_pct_cached = _load_pctmap_from_disk(f"CORE_{view_core}", cfg.window_days, expected_last_date=last_all)
if _core_pct_cached is None:
    try:
        _save_pctmap_to_disk(f"CORE_{view_core}", cfg.window_days, pos_pct, asof_last_date=last_all)
        _core_pct_cached = _load_pctmap_from_disk(f"CORE_{view_core}", cfg.window_days, expected_last_date=last_all)
    except Exception:
        _core_pct_cached = None

if _core_pct_cached is not None and not _core_pct_cached.empty:
    st.caption("Northern Star (this core) RankPos percentiles (cached for stability).")


def _rerun():
    """Compatibility rerun helper (Streamlit versions)."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass