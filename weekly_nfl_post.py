import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types

# --- Configuration ---
SEASON = 2026
TOTAL_WEEKS = 18
STATE_FILE = "last_posted_week.json"
LOGO_DIR = "logos"

PBP_URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.parquet"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# Model Parameters (Locked from Phase 1)
DECAY_RATE = 0.92
SOS_PASSES = 2
Z_CLIP = 2.5
SIGMOID_MIDPOINT = 10.0
SIGMOID_STEEPNESS = 0.5
SIM_TRIALS = 1000
HOME_FIELD_ADV = 5.0

FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.0-flash",
]

TEAM_DIVISIONS = {
    "BUF": ("AFC", "East"), "MIA": ("AFC", "East"), "NE": ("AFC", "East"), "NYJ": ("AFC", "East"),
    "BAL": ("AFC", "North"), "CIN": ("AFC", "North"), "CLE": ("AFC", "North"), "PIT": ("AFC", "North"),
    "HOU": ("AFC", "South"), "IND": ("AFC", "South"), "JAX": ("AFC", "South"), "TEN": ("AFC", "South"),
    "DEN": ("AFC", "West"), "KC": ("AFC", "West"), "LV": ("AFC", "West"), "LAC": ("AFC", "West"),
    "DAL": ("NFC", "East"), "NYG": ("NFC", "East"), "PHI": ("NFC", "East"), "WAS": ("NFC", "East"),
    "CHI": ("NFC", "North"), "DET": ("NFC", "North"), "GB": ("NFC", "North"), "MIN": ("NFC", "North"),
    "ATL": ("NFC", "South"), "CAR": ("NFC", "South"), "NO": ("NFC", "South"), "TB": ("NFC", "South"),
    "ARI": ("NFC", "West"), "LA": ("NFC", "West"), "SF": ("NFC", "West"), "SEA": ("NFC", "West"),
}

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#041E42", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LV": "#000000", "LAC": "#0080C6", "LA": "#003594", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SF": "#AA0000",
    "SEA": "#002244", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}

ESPN_LOGO_MAP = {"LA": "lar", "WAS": "wsh"}

RANKINGS_SYSTEM_PROMPT = """You are @pixelgaragedata (Pixel Garage Graphics), a Principal Data Scientist and NFL analytics nerd who builds clean charts and statistical models.
Write 3 distinct, sharp X post options announcing this week's Tuesday NFL Power Rankings and week over week movement.

Voice and Formatting Rules:
- Sound like an insightful, grounded football analytics practitioner.
- Keep every post strictly under 250 characters so there is plenty of room for the slopegraph chart.
- Do not use hyphens or em dashes anywhere in your text.
- Do not use hashtags or generic filler like "Check this out" or "New rankings dropped".
- Mention specific teams, scores, or rank shifts from the provided summary.

Vary the 3 options:
1. Top Tier Focus: Highlight the #1 team and how much separation exists at the top of the league.
2. Biggest Risers Focus: Highlight the biggest week over week risers and why recent efficiency lifted them.
3. Biggest Fallers or Record vs EPA Divergence Focus: Highlight a team sliding down or a team whose underlying EPA tells a different story than its record.

Return ONLY valid JSON:
{
  "options": [
    "First post option here",
    "Second post option here",
    "Third post option here"
  ]
}
"""

PLAYOFFS_SYSTEM_PROMPT = """You are @pixelgaragedata (Pixel Garage Graphics), a Principal Data Scientist and NFL analytics nerd who builds clean charts and statistical models.
Write 3 distinct, sharp X post options announcing this week's Thursday NFL Playoff and Division Probability projections based on 1,000 full season simulations.

Voice and Formatting Rules:
- Sound like an insightful, grounded football analytics practitioner.
- Keep every post strictly under 250 characters so there is plenty of room for the playoff odds chart.
- Do not use hyphens or em dashes anywhere in your text.
- Do not use hashtags or generic filler like "Check this out" or "New projections dropped".
- Mention specific playoff percentages, division title odds, or #1 seed probabilities from the provided summary.

Vary the 3 options:
1. Tightest Division Race Focus: Highlight a close division battle where two contenders are splitting simulation outcomes.
2. Top Seed & Locks Focus: Highlight the teams dominating conference #1 seed and playoff lock probabilities.
3. Wild Card Bubble Focus: Highlight teams fighting on the playoff cut line heading into the weekend slate.

Return ONLY valid JSON:
{
  "options": [
    "First post option here",
    "Second post option here",
    "Third post option here"
  ]
}
"""


# --- State & Logo Loaders ---
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"season": SEASON, "rankings_week": 0, "playoffs_week": 0}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Migrate legacy single-week format seamlessly
            legacy = int(data.get("last_week", 0))
            return {
                "season": SEASON,
                "rankings_week": int(data.get("rankings_week", legacy)),
                "playoffs_week": int(data.get("playoffs_week", 0)),
            }
    except Exception:
        return {"season": SEASON, "rankings_week": 0, "playoffs_week": 0}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


_LOGO_IMAGE_CACHE: dict[str, Image.Image] = {}


def get_team_logo(team: str, zoom: float = 0.35) -> OffsetImage | None:
    if team not in _LOGO_IMAGE_CACHE:
        local_path = os.path.join(LOGO_DIR, f"{team}.png")
        try:
            if os.path.exists(local_path):
                img = Image.open(local_path).convert("RGBA")
            else:
                code = ESPN_LOGO_MAP.get(team, team.lower())
                url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGBA")
            img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            _LOGO_IMAGE_CACHE[team] = img
        except Exception as e:
            print(f"Warning: could not load logo for {team}: {e}")
            return None

    return OffsetImage(_LOGO_IMAGE_CACHE[team], zoom=zoom)


# --- Core Power Ranking Engine (Phase 1 v2) ---
def calculate_sigmoid_weight_v2(week: int, total_weeks: int = TOTAL_WEEKS) -> float:
    if week >= total_weeks:
        return 1.0
    weight = 1.0 / (1.0 + np.exp(-SIGMOID_STEEPNESS * (week - SIGMOID_MIDPOINT)))
    return float(np.clip(weight, 0.0, 1.0))


def zscore_to_100(series: pd.Series, invert: bool = False) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(50.0, index=series.index)
    z = (series - series.mean()) / std
    if invert:
        z = -z
    z_clipped = np.clip(z, -Z_CLIP, Z_CLIP)
    return 50.0 + (z_clipped / Z_CLIP) * 50.0


def weighted_mean(df: pd.DataFrame, group_cols: list[str], value_col: str, weight_col: str) -> pd.Series:
    temp = df[group_cols].copy()
    temp["_wx"] = df[value_col] * df[weight_col]
    temp["_w"] = df[weight_col]
    agg = temp.groupby(group_cols)[["_wx", "_w"]].sum()
    return agg["_wx"] / agg["_w"]


def compute_rankings_v2(pbp_df: pd.DataFrame, target_week: int) -> pd.DataFrame:
    pbp_wk = pbp_df[pbp_df["week"] <= target_week].copy()

    is_garbage = (pbp_wk["qtr"] >= 4) & ((pbp_wk["wp"] > 0.80) | (pbp_wk["wp"] < 0.20))
    comp = pbp_wk[
        (pbp_wk["play_type"].isin(["pass", "run"]))
        & (~is_garbage)
        & (pbp_wk["epa"].notna())
        & (pbp_wk["down"].isin([1, 2]))
    ].copy()

    comp["recency_weight"] = DECAY_RATE ** (target_week - comp["week"])

    off_s = (
        weighted_mean(comp, ["posteam", "play_type"], "epa", "recency_weight")
        .unstack()
        .rename(columns={"pass": "off_pass_epa", "run": "off_rush_epa"})
    )
    def_s = (
        weighted_mean(comp, ["defteam", "play_type"], "epa", "recency_weight")
        .unstack()
        .rename(columns={"pass": "def_pass_epa", "run": "def_rush_epa"})
    )
    base_s = off_s.join(def_s)

    game_pairs = (
        comp[["game_id", "week", "posteam", "defteam"]]
        .drop_duplicates()
        .assign(game_weight=lambda d: DECAY_RATE ** (target_week - d["week"]))
    )

    curr = base_s.copy()
    for _ in range(SOS_PASSES):
        l_off_p, l_off_r = curr["off_pass_epa"].mean(), curr["off_rush_epa"].mean()
        l_def_p, l_def_r = curr["def_pass_epa"].mean(), curr["def_rush_epa"].mean()

        next_rows = []
        for team in base_s.index:
            o_games = game_pairs[game_pairs["posteam"] == team]
            if not o_games.empty:
                w = o_games["game_weight"].values
                opp_defs = o_games["defteam"].values
                opp_def_p = np.average(curr.loc[opp_defs, "def_pass_epa"].values, weights=w)
                opp_def_r = np.average(curr.loc[opp_defs, "def_rush_epa"].values, weights=w)
            else:
                opp_def_p, opp_def_r = l_def_p, l_def_r

            d_games = game_pairs[game_pairs["defteam"] == team]
            if not d_games.empty:
                w = d_games["game_weight"].values
                opp_offs = d_games["posteam"].values
                opp_off_p = np.average(curr.loc[opp_offs, "off_pass_epa"].values, weights=w)
                opp_off_r = np.average(curr.loc[opp_offs, "off_rush_epa"].values, weights=w)
            else:
                opp_off_p, opp_off_r = l_off_p, l_off_r

            next_rows.append({
                "team": team,
                "off_pass_epa": base_s.loc[team, "off_pass_epa"] - (opp_def_p - l_def_p),
                "off_rush_epa": base_s.loc[team, "off_rush_epa"] - (opp_def_r - l_def_r),
                "def_pass_epa": base_s.loc[team, "def_pass_epa"] - (opp_off_p - l_off_p),
                "def_rush_epa": base_s.loc[team, "def_rush_epa"] - (opp_off_r - l_off_r),
            })
        curr = pd.DataFrame(next_rows).set_index("team")

    curr["weighted_off_epa"] = 0.75 * curr["off_pass_epa"] + 0.25 * curr["off_rush_epa"]
    curr["weighted_def_epa"] = 0.75 * curr["def_pass_epa"] + 0.25 * curr["def_rush_epa"]

    curr["z_off_100"] = zscore_to_100(curr["weighted_off_epa"], invert=False)
    curr["z_def_100"] = zscore_to_100(curr["weighted_def_epa"], invert=True)
    curr["advanced_epa_score"] = 0.60 * curr["z_off_100"] + 0.40 * curr["z_def_100"]

    games_wk = (
        pbp_wk.dropna(subset=["home_score", "away_score"])
        .drop_duplicates(subset=["game_id"], keep="last")
        .copy()
    )
    games_wk["home_win"] = (games_wk["home_score"] > games_wk["away_score"]).astype(int)
    games_wk["away_win"] = (games_wk["away_score"] > games_wk["home_score"]).astype(int)
    games_wk["is_tie"] = (games_wk["home_score"] == games_wk["away_score"]).astype(int)

    h_st = games_wk.groupby("home_team").agg(
        h_w=("home_win", "sum"), h_l=("away_win", "sum"), h_t=("is_tie", "sum"), h_g=("game_id", "count")
    )
    a_st = games_wk.groupby("away_team").agg(
        a_w=("away_win", "sum"), a_l=("home_win", "sum"), a_t=("is_tie", "sum"), a_g=("game_id", "count")
    )
    st_wk = h_st.join(a_st, how="outer").fillna(0)
    st_wk["wins"] = (st_wk["h_w"] + st_wk["a_w"]).astype(int)
    st_wk["losses"] = (st_wk["h_l"] + st_wk["a_l"]).astype(int)
    st_wk["ties"] = (st_wk["h_t"] + st_wk["a_t"]).astype(int)
    st_wk["games_played"] = st_wk["h_g"] + st_wk["a_g"]
    st_wk["win_pct"] = (st_wk["wins"] + 0.5 * st_wk["ties"]) / st_wk["games_played"]
    st_wk = st_wk[["wins", "losses", "ties", "win_pct"]].reset_index().rename(columns={"home_team": "team"})

    rec_w = calculate_sigmoid_weight_v2(target_week)
    epa_w = 1.0 - rec_w

    final_df = curr.reset_index().merge(st_wk, on="team")
    final_df["proprietary_score"] = (epa_w * final_df["advanced_epa_score"]) + (rec_w * final_df["win_pct"] * 100.0)
    final_df = final_df.sort_values(by="proprietary_score", ascending=False).reset_index(drop=True)
    final_df["rank"] = final_df.index + 1
    return final_df


# --- Monte Carlo Playoff Simulation ---
def simulate_playoffs(rankings_df: pd.DataFrame, schedule_df: pd.DataFrame, current_week: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ratings = dict(zip(rankings_df["team"], rankings_df["proprietary_score"]))
    base_wins = dict(zip(rankings_df["team"], rankings_df["wins"] + 0.5 * rankings_df["ties"]))
    div_wins_base = {t: 0.0 for t in ratings}
    conf_wins_base = {t: 0.0 for t in ratings}

    completed = schedule_df[(schedule_df["week"] <= current_week) & (schedule_df["result"].notna())]
    remaining = schedule_df[(schedule_df["week"] > current_week) & (schedule_df["week"] <= TOTAL_WEEKS)]

    for _, g in completed.iterrows():
        h, a, res = g["home_team"], g["away_team"], g["result"]
        if h not in ratings or a not in ratings:
            continue
        same_div = TEAM_DIVISIONS[h] == TEAM_DIVISIONS[a]
        same_conf = TEAM_DIVISIONS[h][0] == TEAM_DIVISIONS[a][0]
        if res > 0:
            if same_div:
                div_wins_base[h] += 1.0
            if same_conf:
                conf_wins_base[h] += 1.0
        elif res < 0:
            if same_div:
                div_wins_base[a] += 1.0
            if same_conf:
                conf_wins_base[a] += 1.0
        else:
            if same_div:
                div_wins_base[h] += 0.5
                div_wins_base[a] += 0.5
            if same_conf:
                conf_wins_base[h] += 0.5
                conf_wins_base[a] += 0.5

    playoff_counts = {t: 0 for t in ratings}
    div_counts = {t: 0 for t in ratings}
    one_seed_counts = {t: 0 for t in ratings}
    rem_records = remaining[["home_team", "away_team"]].to_dict("records")

    for _ in range(SIM_TRIALS):
        w = base_wins.copy()
        dw = div_wins_base.copy()
        cw = conf_wins_base.copy()

        for g in rem_records:
            h, a = g["home_team"], g["away_team"]
            diff = (ratings[h] + HOME_FIELD_ADV) - ratings[a]
            p_home = 1.0 / (1.0 + np.exp(-0.08 * diff))
            same_div = TEAM_DIVISIONS[h] == TEAM_DIVISIONS[a]
            same_conf = TEAM_DIVISIONS[h][0] == TEAM_DIVISIONS[a][0]

            if rng.random() < p_home:
                w[h] += 1.0
                if same_div:
                    dw[h] += 1.0
                if same_conf:
                    cw[h] += 1.0
            else:
                w[a] += 1.0
                if same_div:
                    dw[a] += 1.0
                if same_conf:
                    cw[a] += 1.0

        for conf in ("AFC", "NFC"):
            div_winners = []
            wild_cards = []
            for div in ("East", "North", "South", "West"):
                div_teams = [t for t, (c, d) in TEAM_DIVISIONS.items() if c == conf and d == div]
                div_teams.sort(key=lambda t: (w[t], dw[t], cw[t], ratings[t]), reverse=True)
                winner = div_teams[0]
                div_winners.append(winner)
                div_counts[winner] += 1
                wild_cards.extend(div_teams[1:])

            div_winners.sort(key=lambda t: (w[t], cw[t], ratings[t]), reverse=True)
            one_seed_counts[div_winners[0]] += 1

            wild_cards.sort(key=lambda t: (w[t], cw[t], ratings[t]), reverse=True)
            for t in div_winners + wild_cards[:3]:
                playoff_counts[t] += 1

    sim_rows = []
    for t in ratings:
        sim_rows.append({
            "team": t,
            "playoff_pct": 100.0 * playoff_counts[t] / SIM_TRIALS,
            "div_pct": 100.0 * div_counts[t] / SIM_TRIALS,
            "one_seed_pct": 100.0 * one_seed_counts[t] / SIM_TRIALS,
        })
    return pd.DataFrame(sim_rows).sort_values(by=["playoff_pct", "div_pct"], ascending=False).reset_index(drop=True)


# --- Visualization Generators ---
def render_slopegraph(prev_df: pd.DataFrame, curr_df: pd.DataFrame, prev_wk: int, curr_wk: int, out_path: str) -> pd.DataFrame:
    merged = (
        curr_df[["team", "rank", "wins", "losses", "ties", "proprietary_score"]]
        .rename(columns={"rank": "curr_rank"})
        .merge(prev_df[["team", "rank"]].rename(columns={"rank": "prev_rank"}), on="team")
    )
    merged["delta"] = merged["prev_rank"] - merged["curr_rank"]
    top_risers = set(merged.nlargest(3, "delta")["team"])

    fig, ax = plt.subplots(figsize=(11, 13.5), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for _, row in merged.iterrows():
        t = row["team"]
        y1, y2 = row["prev_rank"], row["curr_rank"]
        is_riser = t in top_risers and row["delta"] > 0
        color = TEAM_COLORS.get(t, "#222222") if is_riser else "#B8BEC6"
        lw = 3.0 if is_riser else 1.2
        alpha = 0.95 if is_riser else 0.55
        zorder = 5 if is_riser else 2

        ax.plot([0, 1], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=zorder)

        logo_left = get_team_logo(t, zoom=0.35)
        if logo_left:
            ax.add_artist(AnnotationBbox(logo_left, (-0.045, y1), frameon=False, zorder=zorder + 1))
        ax.text(-0.10, y1, f"#{y1} {t}", ha="right", va="center", fontsize=9.5,
                fontweight="bold" if is_riser else "normal", color="#111111" if is_riser else "#555555")

        logo_right = get_team_logo(t, zoom=0.35)
        if logo_right:
            ax.add_artist(AnnotationBbox(logo_right, (1.045, y2), frameon=False, zorder=zorder + 1))

        rec_str = f"{row['wins']}-{row['losses']}" + (f"-{row['ties']}" if row["ties"] else "")
        delta_str = f" (+{row['delta']})" if row["delta"] > 0 else (f" ({row['delta']})" if row["delta"] < 0 else " (-)")
        ax.text(1.10, y2, f"#{y2} {t} ({rec_str}){delta_str}", ha="left", va="center", fontsize=9.5,
                fontweight="bold" if is_riser else "normal", color="#111111" if is_riser else "#444444")

    ax.set_ylim(33.2, 0.2)
    ax.set_xlim(-0.34, 1.48)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Week {prev_wk}", f"Week {curr_wk}"], fontsize=13, fontweight="bold", color="#111111")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.suptitle(f"NFL Power Rankings: Week {prev_wk} to Week {curr_wk}",
                 fontsize=17, fontweight="bold", y=0.96, color="#111111")
    ax.set_title(f"{SEASON} Season | @pixelgaragedata", fontsize=11, color="#555555", pad=22)

    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return merged


def render_playoff_chart(sim_df: pd.DataFrame, rankings_df: pd.DataFrame, curr_wk: int, out_path: str) -> None:
    top16 = sim_df.head(16).merge(rankings_df[["team", "wins", "losses", "ties"]], on="team").iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 9.5), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    y_pos = np.arange(len(top16))
    colors = [TEAM_COLORS.get(t, "#2B5B84") for t in top16["team"]]

    ax.barh(y_pos, top16["playoff_pct"], height=0.62, color=colors, alpha=0.88, label="Make Playoffs %")
    ax.barh(y_pos, top16["div_pct"], height=0.62, color="#FFFFFF", alpha=0.35, edgecolor="#111111",
            linewidth=0.8, hatch="//", label="Win Division %")

    for idx, (_, row) in enumerate(top16.iterrows()):
        t = row["team"]
        rec_str = f"{row['wins']}-{row['losses']}" + (f"-{row['ties']}" if row["ties"] else "")
        logo = get_team_logo(t, zoom=0.35)
        if logo:
            ax.add_artist(AnnotationBbox(logo, (-4.5, idx), frameon=False))
        ax.text(-9.5, idx, f"{t} ({rec_str})", ha="right", va="center", fontsize=10, fontweight="bold", color="#111111")
        ax.text(row["playoff_pct"] + 1.5, idx, f"{row['playoff_pct']:.1f}%  (Div: {row['div_pct']:.0f}%)",
                ha="left", va="center", fontsize=9.5, fontweight="bold", color="#222222")

    ax.set_xlim(-26, 118)
    ax.set_yticks([])
    ax.set_xlabel("Simulated Probability (%)", fontsize=11, fontweight="bold", color="#222222")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#888888")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    plt.suptitle(f"NFL Playoff & Division Odds Through Week {curr_wk}",
                 fontsize=16.5, fontweight="bold", y=0.96, color="#111111")
    ax.set_title(f"{SEASON} Season ({SIM_TRIALS:,} Simulations) | @pixelgaragedata",
                 fontsize=11, color="#555555", pad=16)

    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# --- Gemini Copy & GitHub Issue Draft Creator ---
def generate_post_options(gemini_client: genai.Client, system_prompt: str, summary_payload: str) -> list[str]:
    last_err = None
    for model_name in FALLBACK_MODELS:
        for _ in range(3):
            try:
                resp = gemini_client.models.generate_content(
                    model=model_name,
                    contents=summary_payload,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.65,
                    ),
                )
                parsed = json.loads(resp.text)
                opts = [o.strip()[:265] for o in parsed.get("options", []) if isinstance(o, str) and o.strip()]
                if len(opts) >= 3:
                    return opts[:3]
            except Exception as e:
                last_err = f"[{model_name}] {e}"
                print(last_err)
                if "503" in str(e) or "429" in str(e):
                    time.sleep(15)
                    continue
                break
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


def open_draft_issue(title: str, chart_label: str, chart_path: str, snapshot_title: str, snapshot_md: str, options: list[str]) -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    branch = "main"
    chart_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{chart_path}"

    body = (
        f"### {title}\n\n"
        f"![{chart_label}]({chart_url})\n\n"
        f"### {snapshot_title}\n{snapshot_md}\n\n"
        f"---\n\n"
        f"### Post Caption Options for @pixelgaragedata\n"
        f"**1:** {options[0]}\n\n"
        f"**2:** {options[1]}\n\n"
        f"**3:** {options[2]}\n\n"
        f"---\n"
        f"Comment `1`, `2`, `3`, or custom text to publish this post with the chart to X. Comment `skip` to close."
    )

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={
            "title": title,
            "body": body,
            "labels": ["nfl-post-draft"],
        },
        timeout=30,
    )
    resp.raise_for_status()
    print(f"Opened NFL Draft Issue #{resp.json()['number']} ({title})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rankings", "playoffs"], required=True)
    args = parser.parse_args()

    print(f"Running NFL pipeline in '{args.mode}' mode for {SEASON}...")
    pbp = pd.read_parquet(PBP_URL)
    pbp_reg = pbp[pbp["season_type"] == "REG"].copy()

    completed_games = pbp_reg.dropna(subset=["home_score", "away_score"])
    latest_week = int(completed_games["week"].max())
    state = load_state()
    state_key = "rankings_week" if args.mode == "rankings" else "playoffs_week"

    if latest_week <= state[state_key] and os.environ.get("FORCE_RUN") != "true":
        print(f"Week {latest_week} already processed for {args.mode} (saved={state[state_key]}). Exiting.")
        return

    prev_week = max(1, latest_week - 1)
    curr_df = compute_rankings_v2(pbp_reg, latest_week)
    os.makedirs("charts", exist_ok=True)
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    if args.mode == "rankings":
        prev_df = compute_rankings_v2(pbp_reg, prev_week)
        slope_path = f"charts/week_{latest_week:02d}_slopegraph.png"
        merged = render_slopegraph(prev_df, curr_df, prev_week, latest_week, slope_path)

        top10_lines = []
        for _, r in curr_df.head(10).iterrows():
            rec = f"{r['wins']}-{r['losses']}" + (f"-{r['ties']}" if r["ties"] else "")
            top10_lines.append(f"{r['rank']}. {r['team']} ({rec}) | Score: {r['proprietary_score']:.1f} | EPA: {r['advanced_epa_score']:.1f}")
        top10_md = "\n".join(top10_lines)

        risers = merged.nlargest(3, "delta")[["team", "prev_rank", "curr_rank", "delta"]].to_dict("records")
        fallers = merged.nsmallest(3, "delta")[["team", "prev_rank", "curr_rank", "delta"]].to_dict("records")

        payload = (
            f"Season {SEASON} Week {latest_week} Tuesday Power Rankings:\n"
            f"Record Weight: {calculate_sigmoid_weight_v2(latest_week):.1%}\n\n"
            f"Top 10:\n{top10_md}\n\n"
            f"Biggest Risers: {risers}\n"
            f"Biggest Fallers: {fallers}"
        )
        options = generate_post_options(gemini_client, RANKINGS_SYSTEM_PROMPT, payload)
        open_draft_issue(
            title=f"NFL Week {latest_week} Power Rankings Draft",
            chart_label="Slopegraph",
            chart_path=slope_path,
            snapshot_title="Top 10 Power Rankings Snapshot",
            snapshot_md=top10_md,
            options=options,
        )

    else:
        games = pd.read_csv(GAMES_URL)
        sched_2026 = games[(games["season"] == SEASON) & (games["game_type"] == "REG")].copy()
        sim_df = simulate_playoffs(curr_df, sched_2026, latest_week)
        odds_path = f"charts/week_{latest_week:02d}_playoff_odds.png"
        render_playoff_chart(sim_df, curr_df, latest_week, odds_path)

        odds_lines = []
        for idx, r in sim_df.head(14).iterrows():
            odds_lines.append(f"{idx + 1}. {r['team']} | Playoffs: {r['playoff_pct']:.1f}% | Div: {r['div_pct']:.1f}% | #1 Seed: {r['one_seed_pct']:.1f}%")
        odds_md = "\n".join(odds_lines)

        payload = (
            f"Season {SEASON} Through Week {latest_week} Thursday Playoff & Division Odds ({SIM_TRIALS:,} Sims):\n\n"
            f"Top 14 Contenders:\n{odds_md}\n\n"
            f"Full Simulation Table:\n{sim_df.head(20).to_dict('records')}"
        )
        options = generate_post_options(gemini_client, PLAYOFFS_SYSTEM_PROMPT, payload)
        open_draft_issue(
            title=f"NFL Week {latest_week} Playoff & Division Odds Draft",
            chart_label="Playoff Odds",
            chart_path=odds_path,
            snapshot_title="Top 14 Playoff & Division Odds Snapshot",
            snapshot_md=odds_md,
            options=options,
        )

    state[state_key] = latest_week
    save_state(state)


if __name__ == "__main__":
    main()
