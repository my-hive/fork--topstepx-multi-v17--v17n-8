"""v17n-4 experiment 4: fade pre-auction concession in ZN on 2nd-Wednesday auction days."""
import numpy as np
import pandas as pd

FAMILY = "treasury-auction-concession-reversal"
PARAMS = {"stop_pts": 0.25}
# Fixed calendar/time structure, not fitted: 10:00 CT start, 11:59 last observed bar,
# 13:30 CT exit, 2nd Wednesday (day 8-14) as 10y auction-day proxy.
T0, T1, EXIT = 36000, 43140, 48600


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in ("ZN", "TN"):
            o, c, sec = df.open.to_numpy(), df.close.to_numpy(), df.ct_sec.to_numpy()
            for label, idx in df.groupby("session", sort=False).indices.items():
                d = pd.Timestamp(label)
                if d.weekday() != 2 or not 8 <= d.day <= 14:
                    continue
                s = sec[idx]
                a, b = np.flatnonzero(s == T0), np.flatnonzero(s == T1)
                if not len(a) or not len(b):
                    continue
                move = c[idx[b[0]]] - o[idx[a[0]]]
                if move == 0:
                    continue
                side, ref = -int(np.sign(move)), c[idx[b[0]]]
                for j in range(b[0], len(idx)):
                    if s[j] >= EXIT or side * (c[idx[j]] - ref) <= -PARAMS["stop_pts"]:
                        break
                    out[idx[j]] = side
        result[symbol] = out
    return result


def entry_risk(panel):
    return {s: np.full(len(df), 1000 * PARAMS["stop_pts"] + 40.0) for s, df in panel.items()}
