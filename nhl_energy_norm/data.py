
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


SIM_COLUMNS = [

    "Top_Shot_Speed_mph",
    "Max_Skating_Speed_mph",
    "Max_Distance_One_Game_mi",
    "Bursts_Over_20mph_per60",       # derived
    "Total_Distance_Skated_per60",   # derived
    "High_Danger_Shot_Share",        # derived
    "Mid_Danger_Shot_Share",         # derived
    "Long_Danger_Shot_Share",        # derived
]

ROLE_COLUMNS = [
    "Avg_TOI_seconds",
    "Offensive_Zone_Pct",
    "Defensive_Zone_Pct",
]

SYS_AGG_COLUMNS = [
    "Offensive_Zone_Pct",
    "Defensive_Zone_Pct",
]


def _toi_to_seconds(toi_str) -> float:
    if not isinstance(toi_str, str) or ":" not in toi_str:
        return np.nan
    mm, ss = toi_str.split(":")
    return int(mm) * 60 + int(ss)


def load_player_features_csv(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    if "Is_Active" in df.columns:
        df = df[df["Is_Active"] != False] 
    df["Avg_TOI_seconds"] = df["Avg_TOI"].apply(_toi_to_seconds)

    total_toi_hours = (df["Avg_TOI_seconds"] * df["gamesPlayed"]) / 3600.0
    safe_hours = total_toi_hours.replace(0, np.nan)
    df["Bursts_Over_20mph_per60"] = df["Bursts_Over_20mph"] / safe_hours
    df["Total_Distance_Skated_per60"] = df["Total_Distance_Skated_mi"] / safe_hours

    if "All_Shots" in df.columns:
        safe_shots = df["All_Shots"].replace(0, np.nan)
    else:
        safe_shots = safe_hours 

    for zone in ["High_Danger", "Mid_Danger", "Long_Danger"]:
        df[f"{zone}_Shot_Share"] = df[f"{zone}_Shots"] / safe_shots

    if "Neutral_Zone_Pct" not in df.columns:
        if "Offensive_Zone_Pct" in df.columns and "Defensive_Zone_Pct" in df.columns:
            df["Neutral_Zone_Pct"] = 1.0 - df["Offensive_Zone_Pct"] - df["Defensive_Zone_Pct"]
        else:
            df["Neutral_Zone_Pct"] = np.nan

    return df.reset_index(drop=True)


def _impute_median(F: np.ndarray) -> np.ndarray:
    F = F.astype(float).copy()
    for j in range(F.shape[1]):
        col = F[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med if not np.isnan(med) else 0.0
    return F


def build_feature_matrices(df: pd.DataFrame, player_ids: List[int]):
    indexed = df.set_index("Player_ID")
    missing = [pid for pid in player_ids if pid not in indexed.index]
    if missing:
        raise ValueError(
            f"{len(missing)} player_id(s) from the RAPM stint data have no row in "
            f"the features CSV (e.g. {missing[:5]}) -- widen the roster pull or "
            f"drop these players before building K."
        )
    row_order = indexed.loc[player_ids]

    F_sim = _impute_median(row_order[SIM_COLUMNS].to_numpy())
    F_role = _impute_median(row_order[ROLE_COLUMNS].to_numpy())

    team_means = df.groupby("Team")[SYS_AGG_COLUMNS].transform("mean")
    df_with_team_means = df.copy()
    df_with_team_means[[f"{c}_team_mean" for c in SYS_AGG_COLUMNS]] = team_means
    sys_indexed = df_with_team_means.set_index("Player_ID")
    F_sys = _impute_median(
        sys_indexed.loc[player_ids, [f"{c}_team_mean" for c in SYS_AGG_COLUMNS]].to_numpy()
    )

    return F_sim, F_role, F_sys


def load_boxcar_reference(df: pd.DataFrame, player_ids: List[int]) -> pd.DataFrame:
    indexed = df.set_index("Player_ID")
    out = indexed.loc[[pid for pid in player_ids if pid in indexed.index],
                       ["points", "goals", "assists", "gamesPlayed"]].copy()
    toi_sec = indexed.loc[out.index, "Avg_TOI"].apply(_toi_to_seconds)
    total_toi_hr = (toi_sec * indexed.loc[out.index, "gamesPlayed"]) / 3600.0
    out["Points_per_60"] = out["points"] / total_toi_hr.replace(0, np.nan)
    return out



@dataclass
class RapmDataset:
    player_ids: List[int]
    X: np.ndarray
    g: np.ndarray
    stint_seconds: np.ndarray  


def build_rapm_from_game_csv(game_csv_path: str, player_ids: List[int],
                              situation: str = "5on5",
                              target_season: Optional[int] = None,
                              max_duration_mismatch_sec: float = 60.0,
                              use_xg: bool = True,
                              toi_weighted_x: bool = True) -> RapmDataset:


    usecols = ["playerId", "gameId", "playerTeam", "home_or_away", "situation", "icetime"]
    if use_xg:
        usecols.extend(["OnIce_F_xGoals", "OnIce_A_xGoals", "I_F_goals"])  # OnIce pair + fallback
    else:
        usecols.append("I_F_goals")
    if target_season is not None:
        usecols.append("season")

    df = pd.read_csv(game_csv_path, usecols=[c for c in usecols if c != "season"] +
                      (["season"] if target_season is not None else []))
    if target_season is not None:
        df = df[df["season"] == target_season]
    df = df[(df["situation"] == situation) & (df["icetime"] > 0)]

    if use_xg and "OnIce_F_xGoals" not in df.columns:
        print("[build_rapm_from_game_csv] OnIce_F_xGoals not found in CSV, falling back to I_F_goals")
        use_xg = False

    id_to_col = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)
    X_rows, g_rows, dur_rows = [], [], []
    n_mismatch_skipped = 0
    n_no_modeled_players = 0

    for gid, g_df in df.groupby("gameId"):
        home = g_df[g_df["home_or_away"] == "HOME"]
        away = g_df[g_df["home_or_away"] == "AWAY"]
        if home.empty or away.empty:
            continue

        home_dur = home["icetime"].sum() / 5.0
        away_dur = away["icetime"].sum() / 5.0
        if abs(home_dur - away_dur) > max_duration_mismatch_sec:
            n_mismatch_skipped += 1
            continue
        duration = (home_dur + away_dur) / 2.0
        if duration <= 0:
            continue

        row = np.zeros(n)
        if toi_weighted_x:
            for _, pr in home.iterrows():
                col = id_to_col.get(pr["playerId"])
                if col is not None:
                    row[col] = pr["icetime"] / duration
            for _, pr in away.iterrows():
                col = id_to_col.get(pr["playerId"])
                if col is not None:
                    row[col] = -pr["icetime"] / duration
        else:
            for pid in home["playerId"]:
                col = id_to_col.get(pid)
                if col is not None:
                    row[col] = 1.0
            for pid in away["playerId"]:
                col = id_to_col.get(pid)
                if col is not None:
                    row[col] = -1.0
        if not row.any():
            n_no_modeled_players += 1
            continue

        if use_xg:

            home_g = home["OnIce_F_xGoals"].sum() - home["OnIce_A_xGoals"].sum()
            away_g = away["OnIce_F_xGoals"].sum() - away["OnIce_A_xGoals"].sum()
        else:
            home_g = home["I_F_goals"].sum()
            away_g = away["I_F_goals"].sum()

        X_rows.append(row)
        g_rows.append(home_g - away_g)
        dur_rows.append(duration)

    target_label = "OnIce xG diff" if use_xg else "goals"
    x_label = "TOI-weighted" if toi_weighted_x else "binary"
    if n_mismatch_skipped:
        print(f"[build_rapm_from_game_csv] skipped {n_mismatch_skipped} game(s) with "
              f">{max_duration_mismatch_sec}s home/away icetime mismatch")
    if n_no_modeled_players:
        print(f"[build_rapm_from_game_csv] skipped {n_no_modeled_players} game(s) with "
              f"no players in the modeled player_ids universe")
    print(f"[build_rapm_from_game_csv] target: {target_label}, X encoding: {x_label}")

    return RapmDataset(
        player_ids=player_ids,
        X=np.array(X_rows) if X_rows else np.zeros((0, n)),
        g=np.array(g_rows, dtype=float),
        stint_seconds=np.array(dur_rows, dtype=float),
    )



def _get(d: dict, *keys, default=np.nan):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _mmss_to_seconds(t: str) -> float:
    if not t:
        return np.nan
    mm, ss = t.split(":")
    return int(mm) * 60 + int(ss)


PERIOD_LENGTH_SEC = 20 * 60


def _period_to_game_seconds(period: int, elapsed: str) -> float:
    return (period - 1) * PERIOD_LENGTH_SEC + _mmss_to_seconds(elapsed)


def get_regular_season_game_ids(client, season: str) -> List[str]:
    if hasattr(client, "helpers") and hasattr(client.helpers, "get_gameids_by_season"):
        return client.helpers.get_gameids_by_season(season, game_types=[2])
    game_ids = set()
    teams = client.teams.teams_info()
    for t in teams:
        abbr = _get(t, "abbr", "triCode", default=None)
        if abbr is None:
            continue
        sched = client.schedule.get_season_schedule(team_abbr=abbr, season=season)
        games = sched.get("games", sched) if isinstance(sched, dict) else sched
        for g in games:
            gid = _get(g, "id", "gameId")
            if gid:
                game_ids.add(gid)
    return sorted(game_ids)


def _parse_shifts(shift_payload) -> pd.DataFrame:
    shifts = shift_payload.get("data", shift_payload) if isinstance(shift_payload, dict) else shift_payload
    rows = []
    for s in shifts:
        period = _get(s, "period", default=None)
        start = _get(s, "startTime", default=None)
        end = _get(s, "endTime", default=None)
        if period is None or start is None or end is None:
            continue
        rows.append({
            "player_id": _get(s, "playerId"),
            "team_id": _get(s, "teamId"),
            "start_sec": _period_to_game_seconds(period, start),
            "end_sec": _period_to_game_seconds(period, end),
        })
    return pd.DataFrame(rows)


def _parse_goals(pbp_payload, home_team_id, away_team_id) -> pd.DataFrame:
    plays = pbp_payload.get("plays", [])
    rows = []
    for p in plays:
        if _get(p, "typeDescKey", default="") != "goal":
            continue
        period = _get(p.get("periodDescriptor", {}), "number", default=None)
        elapsed = _get(p, "timeInPeriod", default=None)
        scoring_team = _get(p.get("details", {}), "eventOwnerTeamId", default=None)
        if period is None or elapsed is None or scoring_team is None:
            continue
        sign = 1 if scoring_team == home_team_id else (-1 if scoring_team == away_team_id else 0)
        rows.append({"game_sec": _period_to_game_seconds(period, elapsed), "sign": sign})
    return pd.DataFrame(rows)


def build_rapm_stints_live_api(client, game_ids: List[str], player_ids: List[int],
                                pause_sec: float = 0.2) -> RapmDataset:

    id_to_col = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)
    X_rows, g_rows, dur_rows = [], [], []

    for gid in game_ids:
        try:
            box = client.game_center.boxscore(game_id=gid)
            home_id = _get(box.get("homeTeam", {}), "id")
            away_id = _get(box.get("awayTeam", {}), "id")

            shifts = _parse_shifts(client.game_center.shift_chart_data(game_id=gid))
            shifts = shifts[shifts["player_id"].isin(id_to_col)]
            if shifts.empty:
                continue

            pbp = client.game_center.play_by_play(game_id=gid)
            goals = _parse_goals(pbp, home_id, away_id)

            boundaries = sorted(set(shifts["start_sec"]).union(shifts["end_sec"]))
            for t0, t1 in zip(boundaries[:-1], boundaries[1:]):
                if t1 <= t0:
                    continue
                on_ice = shifts[(shifts["start_sec"] <= t0) & (shifts["end_sec"] >= t1)]
                home_skaters = on_ice.loc[on_ice["team_id"] == home_id, "player_id"].tolist()
                away_skaters = on_ice.loc[on_ice["team_id"] == away_id, "player_id"].tolist()
                if len(home_skaters) != 5 or len(away_skaters) != 5:
                    continue

                row = np.zeros(n)
                for pid in home_skaters:
                    row[id_to_col[pid]] = 1.0
                for pid in away_skaters:
                    row[id_to_col[pid]] = -1.0

                stint_goals = goals[(goals["game_sec"] >= t0) & (goals["game_sec"] < t1)]
                diff = float(stint_goals["sign"].sum()) if not stint_goals.empty else 0.0

                X_rows.append(row)
                g_rows.append(diff)
                dur_rows.append(t1 - t0)
        except Exception as e:
            print(f"[build_rapm_stints] skipping game {gid}: {e}")
        time.sleep(pause_sec)

    return RapmDataset(
        player_ids=player_ids,
        X=np.array(X_rows) if X_rows else np.zeros((0, n)),
        g=np.array(g_rows, dtype=float),
        stint_seconds=np.array(dur_rows, dtype=float),
    )