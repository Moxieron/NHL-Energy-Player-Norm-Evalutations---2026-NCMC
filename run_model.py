
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import jax.numpy as jnp

from nhl_energy_norm import model, optimize, graphs, data as nhldata


def _to_jnp(a):
    return jnp.array(a, dtype=jnp.float32)


def run_demo(n_players: int = 60, m_stints: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_x = rng.normal(0, 0.4, size=n_players)

    F_sim = rng.normal(0, 1, size=(n_players, 5)) + true_x[:, None] * 0.3
    F_role = rng.normal(0, 1, size=(n_players, 3)) + true_x[:, None] * 0.15
    F_sys = rng.integers(0, 3, size=(n_players, 2)).astype(float)

    X = rng.choice([-1, 0, 0, 0, 1], size=(m_stints, n_players)).astype(float)
    stint_seconds = rng.uniform(10, 90, size=m_stints)
    noise = rng.normal(0, 0.6, size=m_stints)
    g = X @ true_x + noise
    sigma_inv = stint_seconds / stint_seconds.mean()

    idx = rng.permutation(m_stints)
    split = int(0.7 * m_stints)
    train, val = idx[:split], idx[split:]

    fit = optimize.fit_hyperparameters(
        _to_jnp(X[train]), _to_jnp(g[train]), _to_jnp(sigma_inv[train]),
        _to_jnp(X[val]), _to_jnp(g[val]), _to_jnp(sigma_inv[val]),
        _to_jnp(F_sim), _to_jnp(F_role), _to_jnp(F_sys), n_steps=300, lr=0.05,
    )
    K_final = model.build_K(_to_jnp(F_sim), _to_jnp(F_role), _to_jnp(F_sys), fit.params)
    x_star, A_inv = model.solve_x_star(_to_jnp(X), _to_jnp(g), _to_jnp(sigma_inv), K_final)
    sd = jnp.sqrt(model.posterior_variance(A_inv))

    out = pd.DataFrame({
        "player_id": np.arange(n_players),
        "true_x_synthetic": true_x,
        "x_star": np.array(x_star),
        "x_star_sd": np.array(sd),
    }).sort_values("x_star", ascending=False)

    corr = np.corrcoef(out["x_star"], out["true_x_synthetic"])[0, 1]
    print(out.head(10).to_string(index=False))
    print(f"\ncorrelation with synthetic ground truth: {corr:.3f}")
    print(f"held-out loss: {fit.loss_history[0]:.4f} -> {fit.loss_history[-1]:.4f}")
    out.to_csv("valuations_demo.csv", index=False)
    return out


def run_real(features_csv: str, games_csv: str, roster_csv: str,
             team: str | None, target_season: int | None = None,
             min_games: int = 0, use_xg: bool = True,
             toi_weighted_x: bool = True) -> pd.DataFrame:
    features_df = nhldata.load_player_features_csv(features_csv)

    roster_df = pd.read_csv(roster_csv, usecols=["Player_ID", "First_Name", "Last_Name",
                                                   "Position_Code", "Team"])
    roster_df["Player_Name"] = roster_df["First_Name"] + " " + roster_df["Last_Name"]

    if team:
        features_df = features_df[features_df["Team"] == team].reset_index(drop=True)
    if min_games > 0 and "gamesPlayed" in features_df.columns:
        before = len(features_df)
        features_df = features_df[features_df["gamesPlayed"] >= min_games].reset_index(drop=True)
        print(f"--min-games {min_games}: {before} -> {len(features_df)} players")
    player_ids = features_df["Player_ID"].tolist()
    if not player_ids:
        raise ValueError(f"No players to model (team={team!r}).")
    print(f"Modeling {len(player_ids)} players.")

    print("Building RAPM fit inputs from game CSV...")
    rapm = nhldata.build_rapm_from_game_csv(games_csv, player_ids, target_season=target_season,
                                              use_xg=use_xg, toi_weighted_x=toi_weighted_x)
    m, n = rapm.X.shape
    print(f"{m} game-rows, {n} players (m/n = {m/n:.2f}).")
    if m < 50:
        raise RuntimeError(f"Only {m} rows — too few.")

    F_sim, F_role, F_sys = nhldata.build_feature_matrices(features_df, rapm.player_ids)
    boxcar_ref = nhldata.load_boxcar_reference(features_df, rapm.player_ids)

    positions = features_df.set_index("Player_ID").loc[rapm.player_ids, "Position"].to_numpy()
    pos_mask = graphs.position_group_mask(positions)
    print(f"Position groups: {dict(zip(*np.unique(positions, return_counts=True)))}")

    rng = np.random.default_rng(0)
    sigma_inv = rapm.stint_seconds / rapm.stint_seconds.mean()


    cv = sigma_inv.std() / sigma_inv.mean()



    baseline_loss = float(np.sum(sigma_inv * (rapm.g - rapm.g.mean()) ** 2) / np.sum(sigma_inv))
    print(f"Naive baseline loss (variance of g): {baseline_loss:.4f}")

    print("Fitting hyperparameters (5-fold CV)...")
    fit = optimize.fit_hyperparameters_kfold(
        _to_jnp(rapm.X), _to_jnp(rapm.g), _to_jnp(sigma_inv),
        _to_jnp(F_sim), _to_jnp(F_role), _to_jnp(F_sys), n_folds=5, n_steps=500, lr=0.05,
        sim_group_mask=pos_mask, role_group_mask=pos_mask,
    )
    final_loss = fit.loss_history[-1]
    status = "BELOW" if final_loss < baseline_loss * 0.98 else "AT/ABOVE"
    print(f"Fitted loss: {fit.loss_history[0]:.4f} -> {final_loss:.4f} (baseline: {baseline_loss:.4f}, {status})")

    import jax.nn as _jnn
    p = fit.params
    rp = {
        "alpha": float(_jnn.softplus(p.log_alpha)),
        "beta": float(_jnn.softplus(p.log_beta)),
        "gamma": float(_jnn.softplus(p.log_gamma)),
        "delta": float(_jnn.softplus(p.log_delta)),
        "sigma_sim": float(_jnn.softplus(p.log_sigma_sim)) + 1e-3,
        "sigma_role": float(_jnn.softplus(p.log_sigma_role)) + 1e-3,
        "sigma_sys": float(_jnn.softplus(p.log_sigma_sys)) + 1e-3,
    }
    print("Learned params:", {k: round(v, 4) for k, v in rp.items()})

    K_final = model.build_K(_to_jnp(F_sim), _to_jnp(F_role), _to_jnp(F_sys), fit.params,
                             sim_group_mask=pos_mask, role_group_mask=pos_mask)
    x_star, A_inv = model.solve_x_star(
        _to_jnp(rapm.X), _to_jnp(rapm.g), _to_jnp(sigma_inv), K_final
    )
    sd = jnp.sqrt(model.posterior_variance(A_inv))

    x_np = np.array(x_star)
    sd_np = np.array(sd)
    ratio = x_np.std() / np.median(sd_np)
    print(f"x* spread: std={x_np.std():.4f}, range=[{x_np.min():.4f}, {x_np.max():.4f}]")
    print(f"Median x*_sd: {np.median(sd_np):.4f}, spread/uncertainty ratio: {ratio:.2f}")


    out = pd.DataFrame({
        "player_id": rapm.player_ids,
        "Position": positions,
        "x_star": x_np,
        "x_star_sd": sd_np,
    })

    out = out.merge(roster_df[["Player_ID", "Player_Name", "Position_Code", "Team"]],
                     left_on="player_id", right_on="Player_ID", how="left")

    out = out.merge(boxcar_ref.reset_index(), left_on="player_id", right_on="Player_ID",
                     how="left", suffixes=("", "_boxcar"))

    # rank WITHIN position
    out["x_star_rank"] = out.groupby("Position")["x_star"].rank(ascending=False).astype(int)
    out["pts60_rank"] = out.groupby("Position")["Points_per_60"].rank(ascending=False).astype(int)
    out["rank_divergence"] = out["x_star_rank"] - out["pts60_rank"]

    drop_cols = [c for c in out.columns if c.startswith("Player_ID")]
    out = out.drop(columns=drop_cols)

    out = out.sort_values(["Position", "x_star"], ascending=[True, False])

    out.to_csv("valuations.csv", index=False)


    A_inv_np = np.array(A_inv)

    display_cols = ["Player_Name", "Team", "Position_Code", "x_star", "x_star_sd",
                    "points", "goals", "assists", "gamesPlayed", "Points_per_60",
                    "x_star_rank", "pts60_rank", "rank_divergence"]

    pid_to_col = {pid: i for i, pid in enumerate(rapm.player_ids)}

    for pos, grp in out.groupby("Position"):
        pos_df = grp.sort_values("x_star", ascending=False).reset_index(drop=True)
        pos_filename = f"valuations_{pos}.csv"
        pos_df.to_csv(pos_filename, index=False)

        top = pos_df.head(15)
        bot = pos_df.tail(5)
        print(f"\n{'='*80}")
        print(f"  {pos.upper()} — {len(pos_df)} players")
        print(f"  x* range: [{pos_df['x_star'].min():.3f}, {pos_df['x_star'].max():.3f}]")
        print(f"  median x*_sd: {pos_df['x_star_sd'].median():.3f}")
        print(f"  saved to {pos_filename}")
        print(f"{'='*80}")
        print(f"\n  Top 15:")
        print(top[display_cols].to_string(index=False))
        print(f"\n  Bottom 5:")
        print(bot[display_cols].to_string(index=False))

        top5 = pos_df.head(5)
        if len(top5) >= 2:
            print(f"\n  Pairwise gap significance (top 5 {pos}):")
            print(f"  {'Player A':>25s} vs {'Player B':<25s}  gap     gap_sd  significant?")
            for r1 in range(len(top5)):
                for r2 in range(r1 + 1, len(top5)):
                    row1, row2 = top5.iloc[r1], top5.iloc[r2]
                    i, j = pid_to_col[row1["player_id"]], pid_to_col[row2["player_id"]]
                    gap = float(row1["x_star"] - row2["x_star"])
                    gap_var = float(A_inv_np[i, i] + A_inv_np[j, j] - 2 * A_inv_np[i, j])
                    gap_sd = float(np.sqrt(max(gap_var, 0)))
                    sig = "YES" if abs(gap) > 1.96 * gap_sd else "no"
                    print(f"  {row1['Player_Name']:>25s} vs {row2['Player_Name']:<25s}  "
                          f"{gap:+.3f}  {gap_sd:.3f}   {sig}")

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--features", type=str, default="C:/Program Files/Microsoft VS Code/Data_2025_2026.csv")
    ap.add_argument("--games", type=str, default="C:/Users/Hugh Cho/Downloads/2025/2025.csv")
    ap.add_argument("--roster", type=str, default="C:/Program Files/Microsoft VS Code/NHL_Rosters_2025_2026.csv")
    ap.add_argument("--team", type=str, default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--min-games", type=int, default=0)
    ap.add_argument("--no-xg", action="store_true", help="use raw goals instead of xG")
    ap.add_argument("--no-toi-weight", action="store_true", help="use binary +1/-1 X instead of TOI-weighted")
    args = ap.parse_args()

    run_real(args.features, args.games, args.roster, args.team, args.season,
                 args.min_games, use_xg=not args.no_xg, toi_weighted_x=not args.no_toi_weight)
