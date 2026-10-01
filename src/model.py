from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# used for hyperparameter tuning 
PARAM_GRID = {
   "n_estimators": [100,300,500],
   "max_depth": [3,5,7],
   "learning_rate": [0.01, 0.05, 0.1],
   "subsample": [0.5,0.8,1.0],
   "colsample_bytree":[0.5,0.8,1.0]
}


def save_model(model, path):
    model.save_model(path)

def load_model(path):
    model = XGBRegressor()
    model.load_model(path)
    return model

"""
Train given the feature set from features.py
Uses built in Grid Search CV for hyperparameter tuning 
:returns: the best model 
"""
def train_model(X, y): 

  # Train
  model = XGBRegressor(
      objective="reg:squarederror",
      random_state=42
  )

  grid_search = GridSearchCV(
     estimator=model, 
     param_grid=PARAM_GRID,
     scoring="reg:squarederror",
     cv=5,
     n_jobs=-1,
     verbose=1
  )

  grid_search.fit(X, y)

  print("Best parameters:")
  print(grid_search.best_params_)

  print("Best CV MAE:")
  print(grid_search.best_score_)

  return grid_search.best_estimator_

"""
Print out basic evaluation metrics 
"""
def evaluate_model(model, X_test, y_test): 

  preds = model.predict(X_test)

  mae = mean_absolute_error(y_test, preds)
  rmse = root_mean_squared_error(y_test, preds)

  print("Model results (test):")
  print(f"MAE(mean absolute error): {mae}")
  print(f"RMSE(root mean squared error): {rmse}")
    
    