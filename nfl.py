import nflreadpy as nfl
from src.data import get_team_data, get_schedule_data, get_pbp_data, get_pregame_pbp_stats, get_pregame_team_stats
from src.features import build_feature_set

"""
Holds the code to run the actual program including training the model, getting info, 
displaying info, refreshing the stats etc. 
"""


def main(): 

  features = build_feature_set(2020, 2025)

  print(features.shape)
  print(features.schema)
  print(features.null_count())

if __name__ == "__main__": 
  main()