from rich.console import Console
from rich.table import Table
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

"""
Prints the predictions for the model
"""
def print_predictions(results: pl.DataFrame):
    for row in results.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        spread = row["spread_line"]

        covers = row["home_covers"]
        probability = row["home_cover_probability"]

        if covers == 1:
            prediction = "HOME COVERS"
            prediction_style = "bold green"
        else:
            prediction = "HOME DOES NOT COVER"
            prediction_style = "bold red"

        console.print()

        # Game header
        console.print(
            f"[bold green]{home}[/bold green] "
            f"[dim]@[/dim] "
            f"[bold cyan]{away}[/bold cyan]"
        )

        # Game info
        info = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        info.add_row(
            f"[dim]Spread:[/dim] {spread:+.1f}",
            f"[dim]Confidence:[/dim] {probability:.1%}",
        )
        console.print(info)

        # Prediction
        console.print(
            f"  [dim]Prediction:[/dim] [{prediction_style}]{prediction}[/{prediction_style}]"
        )

"""
Prints the feature set for matchup and stat checking purposes
"""
def print_feature_set(features: pl.DataFrame):
    for row in features.iter_rows(named=True):
        console.print()

        # Game header
        console.print(
            f"[bold green]{row['home_team']}[/bold green] "
            f"[dim]@[/dim] "
            f"[bold cyan]{row['away_team']}[/bold cyan]"
        )

        # Game info
        info = Table(show_header=False, box=None, padding=(0, 2))
        info.add_row(
            f"[dim]Season:[/dim] {row['season']}",
            f"[dim]Week:[/dim] {row['week']}",
            f"[dim]Spread:[/dim] {row['spread']:+.1f}",
        )
        console.print(info)

        # Features
        table = Table(
            title="[bold]Features[/bold]",
            show_header=False,
            box=None,
            padding=(0, 2),
        )

        for column, value in row.items():
            if column in {
                "season",
                "week",
                "home_team",
                "away_team",
                "spread",
                "y",
            }:
                continue

            table.add_row(
                f"[cyan]{column}[/cyan]",
                f"{value:.4f}" if isinstance(value, float) else str(value),
            )

        console.print(table)