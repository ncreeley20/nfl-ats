
"""
Start and stop are inclusive
"""
def generate_season_list(exclude: int, start: int, stop:int): 
  return [n for n in range(start, stop + 1) if n != exclude]
  