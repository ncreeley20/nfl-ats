import nflreadpy as nfl
import polars as pl

def get_schedule_data(season: int, prior_to_week: int): 
   # if prior_to_week = 1 use last season data 
    if prior_to_week == 1: 
       return nfl.load_schedules(seasons=season - 1)
    else: 
      df = nfl.load_schedules(seasons=season)
      return df.filter(pl.col("week") < prior_to_week)

def get_team_data(season: int, prior_to_week: int): 

  # if prior_to_week = 1 use last season data 
  if prior_to_week == 1: 
     return nfl.load_team_stats(seasons=season-1, summary_level="week")
  else: 
    df = nfl.load_team_stats(seasons=season, summary_level="week")
    return df.filter(pl.col("week") < prior_to_week)


def get_pbp_data(season: int, prior_to_week: int):
  # if prior_to_week = 1 use last season data 
  if prior_to_week == 1: 
     return nfl.load_pbp(seasons=season - 1)
  else: 
    df = nfl.load_pbp(seasons=season)
    return df.filter(pl.col("week") < prior_to_week)





