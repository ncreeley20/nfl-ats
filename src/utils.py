
"""
Start and stop are inclusive
"""
def generate_season_list(start: int, stop:int, exclude: int | None = None): 
  return [n for n in range(start, stop + 1) if n != exclude]
  