# Markov Regime-Switching FX Trading System

A 5-layer algorithmic trading system that uses Hidden Markov Models (HMM) to detect market volatility regimes (`low_vol` / `high_vol`) and gates per-symbol strategies accordingly. Built in Python, connected to MetaTrader 5 via the EGM Securities broker.

**Status:** Running on demo. Core architecture validated on 8 years of M15 data across 7 FX pairs. Gold (XAUUSD) added to the signal layer but not yet wired into the data/execution layers.

---

## Overview

The system classifies each currency pair into one of two hidden volatility regimes using a Gaussian HMM, then routes to a regime-appropriate mean-reversion or trend strategy. Each symbol has its own strategy configuration and indicator parameters, discovered through walk-forward backtesting on up to 8 years of 15-minute bar data.

Core thesis: no single strategy works everywhere, but regime-aware strategy switching, combined with per-symbol parameter tuning, produces a more consistent edge than any single static strategy.

---

## Architecture

### Layer 1 — Data Pipeline
Connects to MT5, pulls OHLCV bars every 60 seconds per symbol, engineers HMM input features, and writes them to a local CSV feature store. Includes a market-open guard (`trade_mode > 0`) so symbols are skipped cleanly outside trading hours.

Features computed (identical across Layers 1, 2, and 5 — any mismatch invalidates the model):
- `log_return` — bar-level log return
- `realized_vol` — rolling 20-bar annualised volatility
- `vol_ratio` — 5-bar vol / 20-bar vol (regime-shift signal)
- `atr_pct` — 14-bar ATR normalised by close price
- `bar_range` — (high − low) / close
- `volume_zscore` — z-score of tick volume

### Layer 2 — HMM Regime Detector
Fits a `GaussianHMM` (n_states=2, covariance_type='full') on the feature matrix. States are mapped to `low_vol` / `high_vol` by sorting on mean `realized_vol`. Saves the model, scaler, and regime map as a bundle (`models/{symbol}_hmm.pkl`). Auto-refits weekly via Layer 4.

### Layer 3 — Signal Layer
Reads the HMM bundle, classifies the latest bar, applies a regime filter (`min_confidence=0.65`, `max_switch_prob=0.15`), and runs whichever strategy is configured for that symbol + regime. Returns a `TradeSignal` dataclass with direction, lot size, SL/TP, and a human-readable reason string.

### Layer 4 — Execution Layer
Imports Layer 3 directly. Auto-refits Layer 2 every Monday at 6am via subprocess. Runs pre-trade risk checks (max open positions = number of symbols, 5% equity drawdown halt), sends market orders with SL/TP attached, monitors open positions for regime-change exits, and logs every action to CSV.

### Layer 5 — Validation & Optimisation
Two standalone scripts, neither requires Layers 1–4 running:
- **Validation backtest** — runs the exact production config through an 8-year walk-forward simulation and plots a 3-panel equity/drawdown/win-rate chart per symbol.
- **Optimisation runner** — exhaustively tests every strategy in every regime across every symbol, with resume support so a multi-hour run survives a Ctrl+C.

---

## Key Technical Decisions

- **`step_bars=400`, not 80** — refits the HMM every ~4 trading days instead of every 80 bars. ~5× fewer fits, same walk-forward integrity, and matches the live weekly refit cadence.
- **`n_iter=50`, not 200`** — convergence logs confirmed the EM algorithm was already converged well before 200 iterations on every window tested.
- **Vectorised SL/TP simulation** — replaced the bar-by-bar Python loop with `numpy.argmax` array operations. Identical results, ~50× faster.
- **Resume system** — an MD5 fingerprint of config + setup list is checked on startup; completed runs are skipped, in-progress runs survive interruption.
- **Per-symbol parameters, not shared params** — each symbol gets its own indicator settings (e.g. EURUSD Bollinger std=3.0 vs EURGBP std=2.5), not a one-size-fits-all config.
- **XAUUSD pip conventions** — `pip_size=0.1`, `pip_value_per_lot=1.0` (Gold's conventions differ from FX majors).
- **ATR channel breakout removed from live config** — generated zero trades across every symbol and timeframe tested in the low_vol regime; the M15 window doesn't produce clean channel breaks.

---

## Optimisation Journey

| Version | Data | Symbols | Setups | Key finding |
|---|---|---|---|---|
| v1 | 17 months (~6,000 bars) | 3 | — | GBPUSD EMA_flipped discovered; USDJPY disabled (no edge) |
| v2 | 7 years (~35,000 bars) | 3 | — | Confirmed GBPUSD EMA_flip_10_50 (Sharpe 1.38) and EURUSD BB_20_3.0 (Sharpe 2.10) |
| v3 | 7 years | 7 (added EURGBP, EURCAD, GBPCAD, AUDUSD, USDCAD) | 31 | Per-symbol optimisation; every strategy only tested in its "expected" regime |
| v4 — Exhaustive | 8 years (~175,000 bars) | 8 (added XAUUSD) | ~60 | Every strategy tested in **both** regimes. Runtime 394 minutes. |

### v4 exhaustive findings — the headline discovery

Bollinger Bands and RSI, previously only tested in `high_vol`, turned out to have strong edges in `low_vol` as well:

| Setup | Symbol | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| `bollinger_20_3.0_LOW` | GBPCAD | **5.64** | -3.9% | 70 |
| `rsi_14_20_80_LOW` | EURGBP | 3.71 | -26.8% | 499 |
| `rsi_21_20_80_HIGH` | GBPUSD | 3.68 | -21.3% | 316 |
| `vwap_0.003_LOW` | EURCAD | 4.66 | -15.2% | 326 |
| `ema_flipped_20_100_HIGH` | AUDUSD | 2.73 | -7.5% | 121 |
| `bollinger_20_3.0_LOW` | USDCAD | 2.79 | -6.4% | 88 |
| `ema_flipped_10_200_HIGH` | XAUUSD | 1.33 | -9.3% | 128 |

**Cross-symbol consistency:** `bollinger_20_3.0_LOW` was profitable on 7 of 8 symbols tested — avg Sharpe 2.24, avg max drawdown only -7.8%. The single most robust setup found in the entire project.

---

## Confirmed Validation Results (single-regime baseline)

The first per-symbol config — one strategy per symbol, single regime active — was run through the full 8-year validation backtest and produced these confirmed numbers:

| Symbol | Strategy | Trades | Win Rate | Sharpe | Max DD | P&L |
|---|---|---|---|---|---|---|
| GBPUSD | rsi(21,20,80) high_vol | 315 | 51.4% | 3.35 | -21.3% | $10,237 |
| EURCAD | ema_flipped(20,100) low_vol | 168 | 49.4% | 2.38 | -9.1% | $3,246 |
| EURGBP | bollinger(20,2.5) high_vol | 534 | 46.8% | 1.92 | -19.6% | $10,645 |
| GBPCAD | rsi(14,20,80) high_vol | 689 | 48.8% | 1.75 | -31.1% | $12,305 |
| USDCAD | ema_flipped(20,100) low_vol | 161 | 45.3% | 1.45 | -6.7% | $1,753 |
| AUDUSD | rsi(21,20,80) high_vol | 264 | 48.1% | 1.44 | -18.0% | $2,842 |
| EURUSD | bollinger(20,3.0) high_vol | 159 | 44.0% | 0.95 | -15.6% | $1,065 |

**Combined: 2,290 trades, $42,093.90 total P&L, avg Sharpe 1.89, worst drawdown -31.1% (GBPCAD), best drawdown -6.7% (USDCAD).** Every symbol profitable. Runtime 55m 43s.

---

## Latest Configuration (dual-regime, pending re-validation)

Following the v4 exhaustive results, Layer 3 and Layer 5 were updated to activate **both** regimes per symbol where the data showed edge:

| Symbol | low_vol | high_vol |
|---|---|---|
| EURUSD | rsi(21, 20, 80) | bollinger(20, 3.0) |
| GBPUSD | bollinger(20, 3.0) | rsi(21, 20, 80) |
| EURGBP | rsi(14, 20, 80) | bollinger(20, 2.5) |
| EURCAD | rsi(21, 20, 80) | disabled |
| GBPCAD | bollinger(20, 3.0) | rsi(14, 20, 80) |
| AUDUSD | bollinger(20, 2.5) | ema_flipped(20, 100) |
| USDCAD | bollinger(20, 3.0) | ema_flipped(20, 100) |
| XAUUSD | disabled | ema_flipped(10, 200) — thin sample, monitor only |

**This config has not yet been run through Layer 5's full validation backtest.** The exhaustive optimisation numbers above came from isolated single-strategy test windows, not the combined walk-forward simulation with the regime filter and position sizing logic applied end-to-end. Confirming the combined numbers is the next step before this goes live.

---

## Risk Parameters

| Parameter | Value |
|---|---|
| Base risk per trade | 1% of balance |
| Max risk per trade | 2% of balance |
| Min regime confidence | 0.65 |
| Max regime switch probability | 0.15 |
| SL multiplier | 1.5× ATR |
| Min reward:risk | 1.5:1 |
| Max drawdown halt | 5% equity vs balance |
| Regime-change exit confidence | > 70% |
| Magic number | 20240401 |
| Max open positions | 7 (one per active symbol) |

---

## File Reference

| File | Layer | Purpose |
|---|---|---|
| `markov1.py` | 1 | Data pipeline — 7 FX symbols, EGM Securities terminal |
| `markov2.py` | 2 | HMM model — 7 FX symbols, n_iter=50 |
| `markov3.py` | 3 | Signal layer — 8 symbols incl. XAUUSD, dual-regime config |
| `markov4.py` | 4 | Execution — 7 FX symbols, max_open_positions=7 |
| `markov53.py` | 5 | Validation backtest — 8 symbols, dual-regime config |
| `markov5opt.py` | 5 | Optimisation runner — exhaustive, resume support |

**Terminal path:** `C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe`
**Working directory:** `C:\Users\sir James\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Scripts\Markov\`

**Run order (two terminals):**
```
Terminal 1:  python markov1.py   ← data feed
Terminal 2:  python markov4.py   ← execution (imports Layer 2 + 3 internally)
```

---

## Pending Items

- [ ] Run the full validation backtest on the new dual-regime config to confirm combined performance numbers (current numbers are from isolated optimisation windows, not the end-to-end simulation)
- [ ] Add XAUUSD to Layers 1, 2, and 4 (currently only in Layers 3 and 5) — needs `pip_size=0.1`, `pip_value_per_lot=1.0`, and `max_open_positions` bumped to 8
- [ ] Monitor XAUUSD live once added — sample size is thin (128 trades over 8 years), treat as observation-only until more data accumulates
- [ ] Consider a macro trend filter for EURUSD Bollinger high_vol — the single-symbol 8-year backtest showed a sustained -35.6% drawdown period during the 2020–2022 USD trend cycle
- [ ] Continue monitoring live demo trading (started Monday, April 7) against backtest expectations now that the system has grown from 3 to 7–8 symbols

## Completed Items

- All 5 layers coded, tested, and running on demo
- Market-open guard fixed (`trade_mode > 0`)
- Unicode arrow bug fixed in Layer 2 console output
- MT5 comment field bug fixed (shortened to fit the 31-character limit)
- Standard EMA crossover tested and found weak; flipped EMA (fading the breakout) found to have genuine edge on several pairs
- 8-year exhaustive optimisation completed (394-minute runtime) — discovered Bollinger and RSI both work in `low_vol`, not just `high_vol`
- Resume system with MD5 fingerprint hashing implemented for multi-hour optimisation runs
- Vectorised SL/TP simulation (~50× speed improvement over the bar-by-bar loop)
- `step_bars` increased from 80 to 400 and `n_iter` reduced from 200 to 50 — combined ~16× speedup with no loss of walk-forward integrity

---

## Appendix: Grid/Martingale Scalping EA (separate, exploratory)

A separate, unrelated strategy was evaluated alongside this project: a grid-martingale EA on XAUUSD that scales lot size on adverse retracements and nets the furthest + most recent order at a fixed $1 profit target once 4+ orders are open.

Initial MT5 Strategy Tester output showed a 100% win rate across 27,359 trades, which on inspection is an MT5 reporting artifact — closing basket pairs together logs both legs as "won" even though the individual losing leg is real; it's absorbed into the net basket P&L rather than reported standalone. The equity curve's downside spikes confirm real drawdowns are occurring even though the balance line looks smooth.

The strategy does already use EMA/stochastic trend filters, but the open risk is the unbounded grid depth during a sustained directional move with no retracement — a scenario not well represented in the ~3-year backtest window used. This is **not integrated with the Markov regime detector** and has **not been deployed live**. If pursued further, gating the grid off during HMM-confirmed strong-trend `high_vol` periods would be the natural next step, since that infrastructure already exists in this project.
