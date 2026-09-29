from data import get_schedule_data, get_team_data, get_pbp_data, get_pregame_team_stats, get_pregame_pbp_stats
from utils import generate_season_list
import polars as pl

"""
GOAL: build one dataframe that holds every game back to certain year with each feature for both teams,
with y = home_score - away_score
"""

"""
Builds the feature table enriching it with the pre-game features,
:start: inclusive start year 
:stop: inclusive end year 
"""
def build_feature_set(start: int, stop: int):

  seasons = generate_season_list(start, stop)

  # get all raw weekly data 
  schedule = get_schedule_data(seasons)
  pbp = get_pbp_data(seasons)
  team_stats = get_team_data(seasons)

  feature_rows = []
  for game in schedule.iter_rows(named=True):
    # build out each row 
    home_team = game["home_team"]
    away_team = game["away_team"]
    div_game = game["div_game"] == 1
    week = game["week"]
    season = game["season"]

    # grab all pregame data
    home_pbp_stats = get_pregame_pbp_stats(pbp, home_team, season, week)
    home_team_stats = get_pregame_team_stats(team_stats, home_team, season, week)

    away_pbp_stats = get_pregame_pbp_stats(pbp, away_team, season, week)
    away_team_stats = get_pregame_team_stats(team_stats, away_team, season, week)


    # build out row 
    row = {

      # schedule/identifying features 
      "season": season,
      "week": week,
      "home_team": home_team, 
      "away_team": away_team,
      "spread": game["spread_line"],
      "is_div_game": div_game, 
      "temp": game["temp"],
      "wind": game["wind"],

      # y 
      "y": game["result"],

      # pbp stats
      "home_success_rate": home_pbp_stats["success_rate"][0],
      "home_3d_conversion_rate": home_pbp_stats["3d_conversion_rate"][0],
      "home_redzone_efficency": home_pbp_stats["redzone_efficency"][0],
      "home_def_success_rate": home_pbp_stats["def_success_rate"][0],
      "home_def_epa": home_pbp_stats["def_epa"][0],
      "home_def_rush_epa": home_pbp_stats["def_rush_epa"][0],
      "home_def_pass_epa": home_pbp_stats["def_pass_epa"][0],

      "away_success_rate": away_pbp_stats["success_rate"][0],
      "away_3d_conversion_rate": away_pbp_stats["3d_conversion_rate"][0],
      "away_redzone_efficency": away_pbp_stats["redzone_efficency"][0],
      "away_def_success_rate": away_pbp_stats["def_success_rate"][0],
      "away_def_epa": away_pbp_stats["def_epa"][0],
      "away_def_rush_epa": away_pbp_stats["def_rush_epa"][0],
      "away_def_pass_epa": away_pbp_stats["def_pass_epa"][0],

      # team stats 
      "home_off_passing_epa": home_team_stats["passing_epa"][0],
      "home_off_rushing_epa": home_team_stats["rushing_epa"][0],
      "home_penalty_yards": home_team_stats["penalty_yards"][0],
      "home_turnover_margin": home_team_stats["turnover_margin"][0],
      "home_def_pressures": home_team_stats["def_pressures"][0],

      "away_off_passing_epa": away_team_stats["passing_epa"][0],
      "away_off_rushing_epa": away_team_stats["rushing_epa"][0],
      "away_penalty_yards": away_team_stats["penalty_yards"][0],
      "away_turnover_margin": away_team_stats["turnover_margin"][0],
      "away_def_pressures": away_team_stats["def_pressures"][0],
    }

    feature_rows.append(row)

  return pl.DataFrame(feature_rows)




  