import nflreadpy as nfl
import polars as pl

"""
Grabs schedule data and filters out unneeded cols 
"""
def get_schedule_data(seasons: list[int]): 
  df = nfl.load_schedules(seasons=seasons)

  # filter out postseason games 
  return df.filter(pl.col("game_type") == "REG").select([
    # identifying data 
    "season",
    "week",
    "away_team",
    "home_team",

    # the actual y
    "result",

    # features
    "spread_line",
    "div_game",
    "temp",
    "wind", 
  ])


"""
Grabs all team data week by week, keeping only important columns and 
adds two new columns (turnover margin and pressures)
"""
def get_team_data(seasons: list[int]): 

  team = nfl.load_team_stats(seasons, summary_level="week").filter(pl.col("season_type") == "REG")

  return (
    team
      # add new cols 
      .with_columns(
        [
          (pl.col("def_sacks") + pl.col("def_qb_hits")).alias("def_pressures"),
          (
            # turnovers gained - turnovers lost
            (
              pl.col("fumble_recovery_opp") + 
              pl.col("def_interceptions")
            ) - (
              pl.col("sack_fumbles_lost") + 
              pl.col("receiving_fumbles_lost") + 
              pl.col("rushing_fumbles_lost") + 
              pl.col("passing_interceptions")
            )
          ).alias("turnover_margin")
        ]
      )
      .select(
        [
          "season",
          "week",
          "team",
          "passing_epa",
          "rushing_epa",
          "penalty_yards",
          "turnover_margin",
          "def_pressures"
        ]
      )
  )


"""
Gets all pbp data for given seasons and aggregates it into team, week, season, 
returning features to be used by the model
"""
def get_pbp_data(seasons: list[int]):

  pbp = nfl.load_pbp(seasons).filter((pl.col("season_type") == "REG") & (pl.col("play_deleted") != 1))

  offense = (
    pbp
    .group_by(["season","week","posteam"])
    .agg([

      # offense 
      pl.col("success").mean().alias("success_rate"),
      pl.col("third_down_converted").filter(pl.col("down") == 3).mean().alias("3d_conversion_rate"),
      pl.col("touchdown").filter(pl.col("drive_inside20") == 1).mean().alias("redzone_efficency"),
    ])
    .rename({"posteam":"team"})
  )

  defense = (
    pbp
    .group_by("season", "week", "defteam")
    .agg([
      pl.col("success").mean().alias("def_success_rate"),
      pl.col("epa").mean().alias("def_epa"),
      pl.col("epa").filter(pl.col("play_type") == "run").mean().alias("def_rush_epa"),
      pl.col("epa").filter(pl.col("play_type") == "pass").mean().alias("def_pass_epa")

    ])
    .rename({"defteam": "team"})
  )

  return offense.join(
    defense, 
    on=["season", "week", "team"],
    how="inner"
  )


"""
Given weekly team stats gets the pregame stats for a team and week
NOTE: everything is meaned, unweighted which is incorrect 
ALSO, turnover margin is the only summed column
"""
def get_pregame_team_stats(team_stats, team, season, week):
  prior = team_stats.filter(
    (pl.col("season") == season) & 
    (pl.col("team") == team) &
    (pl.col("week") < week)
  )

  return prior.select([
    pl.col("passing_epa").mean(),
    pl.col("rushing_epa").mean(),
    pl.col("penalty_yards").mean(),
    pl.col("turnover_margin").sum(),
    pl.col("def_pressures").mean()
  ])
 
"""
Given weekly pbp stats gets the pregame stats for a team and week
NOTE: columns are meaned, statistically incorrect and gives more weight to individual games 
"""
def get_pregame_pbp_stats(pbp_stats, team, season, week):
  prior = pbp_stats.filter(
      (pl.col("season") == season) & 
      (pl.col("team") == team) &
      (pl.col("week") < week)
    )
  
  return prior.select([
    pl.col("success_rate").mean(),
    pl.col("3d_conversion_rate").mean(),
    pl.col("redzone_efficency").mean(),
    pl.col("def_success_rate").mean(),
    pl.col("def_epa").mean(),
    pl.col("def_rush_epa").mean(),
    pl.col("def_pass_epa").mean()
  ])









