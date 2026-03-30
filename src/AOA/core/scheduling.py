import numpy as np
import pandas as pd


def optimize_schedule(df, time_col="czas_produkcji_h", alpha=1.0, beta=2.0):
    best_score = float("inf")
    best_df = None

    strategies = {
        "EDF": lambda d: d.sort_values("termin_h"),
        "SPT": lambda d: d.sort_values(time_col),
        "LPT": lambda d: d.sort_values(time_col, ascending=False),
        "Slack": lambda d: d.assign(slack=d["termin_h"] - d[time_col]).sort_values("slack"),
    }

    for name, strategy in strategies.items():
        temp = strategy(df.copy())

        t = 0.0
        lateness_sum = 0.0
        t_start, t_end, lateness = [], [], []

        for _, row in temp.iterrows():
            start = t
            end = start + row[time_col]
            deadline = row["termin_h"]
            late = max(0, end - deadline)

            t_start.append(start)
            t_end.append(end)
            lateness.append(late)
            lateness_sum += late
            t = end

        score = alpha * t + beta * lateness_sum

        if score < best_score:
            best_score = score
            best_df = temp.copy()
            best_df["t_start"] = t_start
            best_df["t_end"] = t_end
            best_df["lateness_h"] = lateness
            best_df["strategy"] = name

    best_df["total_time"] = best_df["t_end"].max()
    best_df["total_lateness"] = best_df["lateness_h"].sum()
    return best_df


def simulate_schedule(df, time_col="czas_produkcji_h"):
    t = 0.0
    t_start, t_end, lateness = [], [], []

    for _, row in df.iterrows():
        start = t
        end = start + row[time_col]
        deadline = row["termin_h"]
        late = max(0, end - deadline)

        t_start.append(start)
        t_end.append(end)
        lateness.append(late)
        t = end

    out = df.copy()
    out["t_start"] = t_start
    out["t_end"] = t_end
    out["lateness_h"] = lateness
    return out


def extract_schedule_features(df):
    return {
        "n_jobs": len(df),
        "mean_time": df["czas_produkcji_h"].mean(),
        "std_time": df["czas_produkcji_h"].std(),
        "max_time": df["czas_produkcji_h"].max(),
        "mean_deadline": df["termin_h"].mean(),
        "load_ratio": df["czas_produkcji_h"].sum() / (df["termin_h"].sum() + 1e-6),
    }


def generate_schedule_label(df, alpha=1.0, beta=2.0):
    best = optimize_schedule(df, alpha=alpha, beta=beta)
    return best["strategy"].iloc[0]
