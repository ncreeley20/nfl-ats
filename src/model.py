from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# NOTE: used for hyperparameter tuning not currently implemented
PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

def save_model(model, path):
    model.save_model(path)

def load_model(path):
    model = XGBClassifier()
    model.load_model(path)
    return model

"""
Train given the feature set from features.py
Uses built in Grid Search CV for hyperparameter tuning 
:returns: the best model 
"""
def train_model(X, y): 

  model = XGBClassifier(
      objective="binary:logistic",
      random_state=42,
      eval_metric="logloss",
  )

  # seasonal data = use time series splits
  cv = TimeSeriesSplit(n_splits=5)

  grid_search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
  )

  grid_search.fit(X, y)

  print("Best parameters:")
  print(grid_search.best_params_)
  print(f"Best CV Accuracy: {grid_search.best_score_:.3%}")
  return grid_search.best_estimator_

"""
Print out basic evaluation metrics and test against dummy classifiers
"""
def evaluate_model(model, X_test, y_test, X_train, y_train): 

  # predict from generated model
  preds = model.predict(X_test)
  probs = model.predict_proba(X_test)[:, 1]

  # set up baselines
  most_freq = DummyClassifier(strategy="most_frequent")
  uniform = DummyClassifier(strategy="uniform")

  most_freq.fit(X_train, y_train)
  uniform.fit(X_train, y_train)

  most_freq_preds = most_freq.predict(X_test)
  uniform_preds = uniform.predict(X_test)

  most_freq_acc = accuracy_score(y_test,most_freq_preds)
  uniform_acc = accuracy_score(y_test, uniform_preds)

  print(f"------------------------- Baseline Results (Dummy) -------------------------")
  print(f"Most Frequent Accuracy (Class Imbalance): {most_freq_acc:.3%}")
  print(f"Uniform Accuracy (Pure 50/50 Predictor): {uniform_acc:.3%}")
  print("------------------------- Model results (test) -------------------------")
  print(f"Accuracy: {accuracy_score(y_test, preds):.3%}")
  print(f"Log Loss: {log_loss(y_test, probs):.4f}")
  print(f"AUC: {roc_auc_score(y_test, probs):.4f}")
    
    