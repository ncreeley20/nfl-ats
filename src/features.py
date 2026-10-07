from src.data import get_schedule_data, get_team_data, get_pbp_data, get_pregame_team_stats, get_pregame_pbp_stats
import polars as pl
from src.utils import to_float


IDENTIFYING_FEATURES = [
  "home_team",
  "away_team",
  "season",
]

"""
Builds the feature table enriching it with the pre-game features,
:start: inclusive start year 
:stop: inclusive end year 
:current: if included then only returns the most recent week rows
"""
def build_feature_set(seasons: list[int], current: bool = False):

  # get all raw weekly data 
  schedule = get_schedule_data(seasons)
  pbp = get_pbp_data(seasons)
  team_stats = get_team_data(seasons)

  if current: 
    max_week = schedule["week"].max()
    schedule = schedule.filter(pl.col("week") == max_week)

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

    # get y 
    home_covered = game["result"] + game["spread_line"] > 0

    # build out row 
    row = {

      # schedule/identifying features 
      "season": season,
      "week": week,
      "home_team": home_team, 
      "away_team": away_team,
      "spread": to_float(game["spread_line"]),
      "is_div_game": div_game, 
      # "temp": game["temp"],
      # "wind": game["wind"],

      # y 
      "y": int(home_covered),

      # ============================================================
      # MATCHUP DIFFERENCES
      # Positive = home team advantage
      # ============================================================

      # Offensive performance
      "success_rate_diff": (
          to_float(home_pbp_stats["success_rate"][0])
          - to_float(away_pbp_stats["success_rate"][0])
      ),

      "3d_conversion_rate_diff": (
          to_float(home_pbp_stats["3d_conversion_rate"][0])
          - to_float(away_pbp_stats["3d_conversion_rate"][0])
      ),

      "redzone_efficiency_diff": (
          to_float(home_pbp_stats["redzone_efficency"][0])
          - to_float(away_pbp_stats["redzone_efficency"][0])
      ),

      "off_passing_epa_diff": (
          to_float(home_team_stats["passing_epa"][0])
          - to_float(away_team_stats["passing_epa"][0])
      ),

      "off_rushing_epa_diff": (
          to_float(home_team_stats["rushing_epa"][0])
          - to_float(away_team_stats["rushing_epa"][0])
      ),

      # Defensive performance
      "def_success_rate_diff": (
          to_float(home_pbp_stats["def_success_rate"][0])
          - to_float(away_pbp_stats["def_success_rate"][0])
      ),

      "def_epa_diff": (
          to_float(home_pbp_stats["def_epa"][0])
          - to_float(away_pbp_stats["def_epa"][0])
      ),

      "def_rush_epa_diff": (
          to_float(home_pbp_stats["def_rush_epa"][0])
          - to_float(away_pbp_stats["def_rush_epa"][0])
      ),

      "def_pass_epa_diff": (
          to_float(home_pbp_stats["def_pass_epa"][0])
          - to_float(away_pbp_stats["def_pass_epa"][0])
      ),

      "def_pressures_diff": (
          to_float(home_team_stats["def_pressures"][0])
          - to_float(away_team_stats["def_pressures"][0])
      ),

      # Turnovers
      "turnover_margin_diff": (
          to_float(home_team_stats["turnover_margin"][0])
          - to_float(away_team_stats["turnover_margin"][0])
      ),

      # # Penalties — lower is better, so reverse the subtraction
      # "penalty_yards_advantage": (
      #     to_float(away_team_stats["penalty_yards"][0])
      #     - to_float(home_team_stats["penalty_yards"][0])
      # ),

      # ============================================================
      # OFFENSE VS DEFENSE MATCHUPS
      # Positive = home team matchup advantage
      # ============================================================

      "home_pass_off_vs_away_pass_def": (
          to_float(home_team_stats["passing_epa"][0])
          - to_float(away_pbp_stats["def_pass_epa"][0])
      ),

      "away_pass_off_vs_home_pass_def": (
          to_float(away_team_stats["passing_epa"][0])
          - to_float(home_pbp_stats["def_pass_epa"][0])
      ),

      "home_rush_off_vs_away_rush_def": (
          to_float(home_team_stats["rushing_epa"][0])
          - to_float(away_pbp_stats["def_rush_epa"][0])
      ),

      "away_rush_off_vs_home_rush_def": (
          to_float(away_team_stats["rushing_epa"][0])
          - to_float(home_pbp_stats["def_rush_epa"][0])
      ),
    }

    feature_rows.append(row)

  return pl.DataFrame(feature_rows, infer_schema_length=None).sort(["season", "week"])

"""
Takes the feature set from above and removes atrributes
not needed for training. Splits the data set based on split 
which is an inclusive year
"""
def build_test_train_split(feature_set: pl.DataFrame, split: int): 
  
  train = feature_set.filter(pl.col("season") < split)

  test = feature_set.filter(pl.col("season") >= split)

  # drop out identifying features 
  clean_train = train.drop(IDENTIFYING_FEATURES)
  clean_test = test.drop(IDENTIFYING_FEATURES)

  X_train = clean_train.drop("y")
  y_train = clean_train["y"]

  X_test = clean_test.drop("y")
  y_test = clean_test["y"]

  return X_train, y_train, X_test, y_test

"""
Simple function to get just one split (no test)
"""
def build_training_set(feature_set: pl.DataFrame): 

  cleaned = feature_set.drop(IDENTIFYING_FEATURES)

  X_train = cleaned.drop("y")
  y_train = cleaned["y"]

  return X_train, y_train




  