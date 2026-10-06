import MetaTrader5 as mt5
import pandas as pd
import validation as bt

sym = "EURUSD"
if not mt5.initialize(path=bt.CONFIG["terminal_path"]):
    raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

raw = None
for n in (170000, 100000, 50000, 30000, 20000):
    bt.CONFIG["n_bars"] = n
    raw = bt.fetch_history(sym)
    if raw is not None:
        print(f"OK with n_bars={n}")
        break
    print(f"n_bars={n} failed | last_error={mt5.last_error()}")

if raw is None:
    mt5.shutdown()
    raise SystemExit("No data at any size. Check the symbol name in Market Watch (EURUSD may have a suffix like EURUSD.m).")

df = bt.engineer_features(raw)

tr, te, st = bt.CONFIG["train_bars"], bt.CONFIG["test_bars"], bt.CONFIG["step_bars"]
out = []
for s in range(tr, len(df) - te, st):
    model, scaler, rmap = bt.fit_hmm(df.iloc[s-tr:s])
    if model is None:
        continue
    out.append(bt.classify_window(model, scaler, rmap, df.iloc[s:s+te])[["regime"]])

r = pd.concat(out)
r["hour"] = r.index.hour
counts = pd.crosstab(r["hour"], r["regime"])

print(pd.crosstab(r["hour"], r["regime"], normalize="index").round(2))
purity = counts.max(axis=1).sum() / counts.values.sum()
base = counts.sum().max() / counts.values.sum()
print(f"\nhour purity: {purity:.0%}  |  baseline (always guess bigger regime): {base:.0%}")
mt5.shutdown()