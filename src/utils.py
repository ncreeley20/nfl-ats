from rich.console import Console
import polars as pl

console = Console()


"""
Start and stop are inclusive
"""
def generate_season_list(start: int, stop:int, exclude: int | None = None): 
  return [n for n in range(start, stop + 1) if n != exclude]

"""
To make sure that all values are correctly being set
"""
def to_float(value):
    return float(value) if value is not None else 0

def print_predictions(results: pl.DataFrame):
    for row in results.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        spread = row["spread_line"]

        covers = row["home_covers"]
        probability = row["home_cover_probability"]

        result = "HOME COVERS" if covers == 1 else "HOME DOES NOT COVER"

        console.print()
        console.print(f"[bold]{away}[/bold] @ [bold]{home}[/bold]")
        console.print(f"  Spread: {spread}")
        console.print(f"  Prediction: {result}")
        console.print(f"  Confidence: {probability:.1%}")