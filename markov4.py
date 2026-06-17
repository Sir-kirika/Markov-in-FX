"""
=============================================================
LAYER 4 — EXECUTION LAYER (final)
TradeSignal -> Order Manager -> Risk Manager -> Trade Logger
+ Auto HMM refit scheduler (Layer 2 every Monday 6am)
=============================================================
Requirements:
    pip install MetaTrader5 pandas

Changes from previous version:
    - USDJPY removed from symbols
    - Terminal path updated to EGM Securities
    - Layer 3 import updated to match final filename
    - Regime exit comment parser updated to new format
    - max_open_positions reduced to 2 (only 2 active pairs)
=============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import os
import sys
import time
import subprocess
import importlib.util
from datetime import datetime


# ─────────────────────────────────────────────
# IMPORT LAYER 3
# ─────────────────────────────────────────────

def _import_layer3():
    """
    Import Layer 3 — tries common filenames in order.
    Update the list if you renamed your Layer 3 file.
    """
    for name in ["layer3_signal_layer", "markov3"]:
        try:
            if name in sys.modules:
                return sys.modules[name]
            spec = importlib.util.spec_from_file_location(name, f"{name}.py")
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[name] = mod
            return mod
        except FileNotFoundError:
            continue
    raise ImportError(
        "Could not find Layer 3 file. "
        "Name it layer3_signal_layer.py, markov3.py, or markov31.py"
    )


L3              = _import_layer3()
generate_signal = L3.generate_signal
TradeSignal     = L3.TradeSignal


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "symbols":       ["EURUSD", "GBPUSD"],
    "terminal_path": r"C:\Program Files\EGM Securities MetaTrader 5 Terminal\terminal64.exe",
    "loop_sleep":    60,
    "log_dir":       "logs/",

    # ── Risk controls ─────────────────────────
    "max_open_positions":    2,
    "max_drawdown_pct":      0.05,
    "exit_on_regime_change": True,

    # ── Order settings ────────────────────────
    "deviation":    20,
    "magic_number": 20240401,

    # ── Pip sizes ─────────────────────────────
    "pip_size": {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
    },

    # ── HMM auto-refit ────────────────────────
    "refit": {
        "filename":    "markov2.py",
        "day":         0,
        "hour":        6,
        "last_refit":  None,
        "timeout_sec": 120,
    },
}


# ─────────────────────────────────────────────
# LAYER 2 AUTO-REFIT
# ─────────────────────────────────────────────

def maybe_refit_hmm():
    """
    Refit HMM when any of these are true:
        - First run of session
        - Monday at 6am
        - More than 7 days since last refit
    Runs as subprocess, prints output into this terminal.
    """
    cfg = CONFIG["refit"]
    now = datetime.now()

    never_refit   = cfg["last_refit"] is None
    is_monday_6am = (now.weekday() == cfg["day"] and now.hour == cfg["hour"])
    overdue       = (cfg["last_refit"] is not None and
                     (now - cfg["last_refit"]).days >= 7)

    if not (never_refit or is_monday_6am or overdue):
        return

    reason = ("startup"            if never_refit   else
              "Monday 6am"         if is_monday_6am else
              "7-day safety refit")

    print(f"\n  [HMM REFIT] Triggered ({reason}) — running {cfg['filename']}...")

    try:
        result = subprocess.run(
            ["python", cfg["filename"]],
            capture_output=True,
            text=True,
            timeout=cfg["timeout_sec"],
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"  [HMM REFIT ERROR]\n{result.stderr}")
        else:
            cfg["last_refit"] = now
            print(f"  [HMM REFIT] Done at {now.strftime('%Y-%m-%d %H:%M')}")

    except subprocess.TimeoutExpired:
        print(f"  [HMM REFIT] Timed out after {cfg['timeout_sec']}s")
    except FileNotFoundError:
        print(f"  [HMM REFIT] File not found: {cfg['filename']}")


# ─────────────────────────────────────────────
# MT5 CONNECTION
# ─────────────────────────────────────────────

def connect_mt5() -> bool:
    if not mt5.initialize(path=CONFIG["terminal_path"]):
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        return False
    account = mt5.account_info()
    print(f"[OK] MT5 connected | Account: {account.login} | "
          f"Balance: {account.balance} {account.currency} | "
          f"Leverage: 1:{account.leverage}")
    return True


def get_account_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 10000.0


def get_account_equity() -> float:
    info = mt5.account_info()
    return info.equity if info else 0.0


# ─────────────────────────────────────────────
# POSITION MANAGEMENT
# ─────────────────────────────────────────────

def get_open_positions(symbol: str = None) -> list:
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None:
        return []
    return [p for p in positions if p.magic == CONFIG["magic_number"]]


def has_open_position(symbol: str) -> bool:
    return len(get_open_positions(symbol)) > 0


def get_total_open_count() -> int:
    return len(get_open_positions())


# ─────────────────────────────────────────────
# PRE-TRADE RISK CHECKS
# ─────────────────────────────────────────────

def pre_trade_checks(signal, balance: float) -> tuple:
    if signal.direction == "flat":
        return False, "Signal is flat — no trade"

    if has_open_position(signal.symbol):
        return False, f"Already have open position on {signal.symbol}"

    open_count = get_total_open_count()
    if open_count >= CONFIG["max_open_positions"]:
        return False, f"Max positions reached ({open_count}/{CONFIG['max_open_positions']})"

    equity = get_account_equity()
    dd_pct = (balance - equity) / balance if balance > 0 else 0
    if dd_pct > CONFIG["max_drawdown_pct"]:
        return False, (f"Drawdown limit breached: {dd_pct:.2%} "
                       f"> {CONFIG['max_drawdown_pct']:.2%} — trading halted")

    if signal.lot_size <= 0:
        return False, "Lot size is zero — check position sizing"

    return True, "All checks passed"


# ─────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────

def get_sl_tp_prices(symbol: str, direction: str,
                     sl_pips: float, tp_pips: float) -> tuple:
    tick     = mt5.symbol_info_tick(symbol)
    pip_size = CONFIG["pip_size"].get(symbol, 0.0001)

    if direction == "buy":
        entry = tick.ask
        sl    = round(entry - sl_pips * pip_size, 5)
        tp    = round(entry + tp_pips * pip_size, 5)
    else:
        entry = tick.bid
        sl    = round(entry + sl_pips * pip_size, 5)
        tp    = round(entry - tp_pips * pip_size, 5)

    return sl, tp


def send_order(signal) -> dict:
    symbol    = signal.symbol
    direction = signal.direction
    lot_size  = signal.lot_size

    info = mt5.symbol_info(symbol)
    if info is None:
        return {"success": False, "error": f"Symbol {symbol} not found"}
    if not info.visible:
        mt5.symbol_select(symbol, True)

    sl_price, tp_price = get_sl_tp_prices(
        symbol, direction, signal.stop_loss_pips, signal.take_profit_pips
    )

    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
    tick       = mt5.symbol_info_tick(symbol)
    price      = tick.ask if direction == "buy" else tick.bid

    # Short ASCII comment — avoids MT5 comment validation errors
    # Format: "HMM high boll" or "HMM low emaf"
    comment = f"HMM {signal.regime[:4]} {signal.strategy[:4]}"

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot_size,
        "type":         order_type,
        "price":        price,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    CONFIG["deviation"],
        "magic":        CONFIG["magic_number"],
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        return {"success": False, "error": str(mt5.last_error())}
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "retcode": result.retcode,
                "error": result.comment}

    return {
        "success":  True,
        "order_id": result.order,
        "price":    result.price,
        "volume":   result.volume,
        "sl":       sl_price,
        "tp":       tp_price,
        "comment":  comment,
    }


# ─────────────────────────────────────────────
# CLOSE POSITION
# ─────────────────────────────────────────────

def close_position(position, reason: str = "") -> dict:
    symbol     = position.symbol
    ticket     = position.ticket
    volume     = position.volume
    pos_type   = position.type
    close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
    tick       = mt5.symbol_info_tick(symbol)
    price      = tick.bid if pos_type == 0 else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    CONFIG["deviation"],
        "magic":        CONFIG["magic_number"],
        "comment":      f"CLOSE {reason[:20]}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        error = result.comment if result else str(mt5.last_error())
        return {"success": False, "ticket": ticket, "error": error}

    return {"success": True, "ticket": ticket, "price": result.price}


# ─────────────────────────────────────────────
# REGIME CHANGE EXIT MONITOR
# ─────────────────────────────────────────────

def check_regime_exits(signals: dict):
    """
    Parse regime from order comment and compare to current regime.
    Comment format: "HMM high boll" or "HMM low emaf"
    parts[1] = "high" or "low" -> maps to high_vol / low_vol
    """
    positions = get_open_positions()

    for pos in positions:
        symbol  = pos.symbol
        comment = pos.comment

        try:
            parts           = comment.split()
            regime_prefix   = parts[1]
            original_regime = "high_vol" if regime_prefix == "high" else "low_vol"
        except IndexError:
            continue

        current_signal = signals.get(symbol)
        if current_signal is None:
            continue

        current_regime = current_signal.regime

        if (CONFIG["exit_on_regime_change"]
                and current_regime != original_regime
                and current_signal.confidence > 0.70):

            print(f"  [REGIME SHIFT] {symbol}: {original_regime} -> {current_regime} "
                  f"| Closing #{pos.ticket}")
            result = close_position(pos, reason=f"regime {current_regime[:4]}")
            log_action("CLOSE_REGIME_SHIFT", symbol, result, current_signal)


# ─────────────────────────────────────────────
# TRADE LOGGER
# ─────────────────────────────────────────────

def log_action(action: str, symbol: str, result: dict, signal=None):
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    log_path = os.path.join(CONFIG["log_dir"], "trade_log.csv")

    row = {
        "timestamp":   datetime.now().isoformat(),
        "action":      action,
        "symbol":      symbol,
        "direction":   signal.direction   if signal else "",
        "regime":      signal.regime      if signal else "",
        "confidence":  signal.confidence  if signal else 0.0,
        "strategy":    signal.strategy    if signal else "",
        "lot_size":    signal.lot_size    if signal else 0.0,
        "sl_pips":     signal.stop_loss_pips  if signal else 0.0,
        "tp_pips":     signal.take_profit_pips if signal else 0.0,
        "success":     result.get("success", False),
        "order_id":    result.get("order_id", ""),
        "exec_price":  result.get("price", ""),
        "error":       result.get("error", ""),
        "reason":      signal.reason if signal else result.get("error", ""),
    }

    df_row       = pd.DataFrame([row])
    write_header = not os.path.exists(log_path)
    df_row.to_csv(log_path, mode="a", header=write_header, index=False)


def print_position_summary():
    positions = get_open_positions()
    if not positions:
        print("  [POSITIONS] None open")
        return
    print(f"  [POSITIONS] {len(positions)} open:")
    for p in positions:
        print(f"    #{p.ticket} {p.symbol} "
              f"{'BUY' if p.type == 0 else 'SELL'} "
              f"{p.volume} lots | P&L: {p.profit:+.2f}")


# ─────────────────────────────────────────────
# MAIN EXECUTION LOOP
# ─────────────────────────────────────────────

def run_execution_layer():
    print("=" * 60)
    print("  LAYER 4 — EXECUTION LAYER STARTING")
    print("=" * 60)

    if not connect_mt5():
        print("[FATAL] Cannot connect to MT5.")
        return

    account = mt5.account_info()
    if account.trade_mode != 0:
        print("[WARNING] This appears to be a LIVE account.")
        confirm = input("Type YES to continue, anything else to exit: ")
        if confirm.strip() != "YES":
            print("[ABORTED]")
            mt5.shutdown()
            return

    try:
        cycle = 0
        while True:
            cycle    += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*60}")
            print(f"[CYCLE {cycle}] {timestamp}")
            print(f"{'='*60}")

            # ── 1. Auto-refit HMM ──────────────────
            maybe_refit_hmm()

            # ── 2. Account state ───────────────────
            balance = get_account_balance()
            equity  = get_account_equity()
            print(f"Balance: {balance:.2f} | Equity: {equity:.2f} | "
                  f"Open: {get_total_open_count()}")

            # ── 3. Generate signals ────────────────
            signals = {}
            for symbol in CONFIG["symbols"]:
                signal          = generate_signal(symbol, balance=balance)
                signals[symbol] = signal
                print(f"\n  [{symbol}] {signal.direction.upper()} | "
                      f"{signal.regime} ({signal.confidence:.0%}) | "
                      f"{signal.reason[:80]}")

            # ── 4. Regime exit check ───────────────
            print("\n  [REGIME EXIT CHECK]")
            check_regime_exits(signals)

            # ── 5. Position summary ────────────────
            print_position_summary()

            # ── 6. Process signals -> orders ───────
            print("\n  [ORDER PROCESSING]")
            for symbol, signal in signals.items():
                approved, check_reason = pre_trade_checks(signal, balance)

                if not approved:
                    print(f"  {symbol}: SKIP — {check_reason}")
                    log_action("SKIP", symbol,
                               {"success": False, "error": check_reason}, signal)
                    continue

                print(f"  {symbol}: SENDING {signal.direction.upper()} "
                      f"{signal.lot_size} lots | "
                      f"SL={signal.stop_loss_pips}pip TP={signal.take_profit_pips}pip")

                result = send_order(signal)

                if result["success"]:
                    print(f"  {symbol}: FILLED ✓ | "
                          f"#{result['order_id']} @ {result['price']} | "
                          f"SL={result['sl']} TP={result['tp']}")
                    log_action("OPEN", symbol, result, signal)
                else:
                    print(f"  {symbol}: FAILED ✗ | {result.get('error', 'Unknown')}")
                    log_action("FAILED", symbol, result, signal)

            print(f"\n[SLEEP] Next cycle in {CONFIG['loop_sleep']}s...")
            time.sleep(CONFIG["loop_sleep"])

    except KeyboardInterrupt:
        print("\n[STOPPED] Execution layer stopped.")

    finally:
        mt5.shutdown()
        print("[INFO] MT5 disconnected.")


if __name__ == "__main__":
    run_execution_layer()