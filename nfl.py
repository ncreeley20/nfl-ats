import typer
import polars as pl
from typing import Optional
from rich.console import Console
from src.utils import generate_season_list
from src.features import build_feature_set, build_test_train_split, build_training_set
from src.model import train_model, evaluate_model, save_model, load_model
from src.utils import print_predictions

DEFAULT_MODEL = "models/nfl_ats.json"

# NOTE: UPDATE FOR CURRENT YEAR
NFL_SEASON = 2026

app = typer.Typer()

console = Console()

@app.command(help="Provide start and end year to train and save a model." \
"If test_year provided splits data set by year and provides test/training split evaluation.")
def train(start: int, end: int, model_name:Optional[str] = None, test_year: Optional[int] = None):

  # set up the data 
  seasons = generate_season_list(start, end)

  with console.status("[bold green]Loading features...[/bold green]"):
    feature_set = build_feature_set(seasons)

  model = None

  if test_year is not None: 
    # build with test and training 
    X_train, y_train, X_test, y_test = build_test_train_split(feature_set, test_year)

    with console.status("[bold green]Training model...[/bold green]"):
      model = train_model(X_train, y_train)

    # test model 
    with console.status("[bold green]Evaluating model...[/bold green]"):
      evaluate_model(model, X_test, y_test, X_train, y_train)

  else: 
    # just build with test
    X, y = build_training_set(feature_set)

    with console.status("[bold green]Training model...[/bold green]"):
      model = train_model(X, y)

  # save model
  path = DEFAULT_MODEL if model_name is None else f"models/{model_name}" 
  save_model(model, path)
  console.print(f"[bold green]Model successfully saved to {path}[/bold green]")


@app.command(help="Predict the current week for most recent NFL season.")
def predict(model_name: Optional[str] = None):

  path = DEFAULT_MODEL if model_name is None else f"models/{model_name}"

  with console.status("[bold green]Loading model...[/bold green]"):
    model = load_model(path)

  with console.status("[bold green]Predicting games...[/bold green]"):
    features = build_feature_set([NFL_SEASON], current=True)
    X, _ = build_training_set(features)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

  results = features.with_columns([
    pl.Series("home_covers", predictions),
    pl.Series("home_cover_probability", probabilities),
  ])

  # print results 
  print(f"//////////////////////////////////////// Predictions for Week {results["week"][0]} ////////////////////////////////////////")
  print_predictions(results)


if __name__ == "__main__": 
  app()