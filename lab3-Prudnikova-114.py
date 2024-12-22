"""
Прудникова Анастасия М8О-114СВ-24
Лабораторная работа №3

Classification; dataset - Breast Cancer; Random Forest

Запуск контейнера:
docker run --name postgres-optuna -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15.5
"""

import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from sklearn.metrics import accuracy_score

from optuna.visualization import (
    plot_optimization_history,
    plot_slice,
    plot_parallel_coordinate,
)

def objective(trial):
    dataset = sklearn.datasets.load_breast_cancer()
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42
    )

    n_trees = trial.suggest_int("n_trees", 10, 200)
    tree_depth = trial.suggest_int("tree_depth", 2, 32, log=True)
    min_samples = trial.suggest_int("min_samples", 2, 20)

    classifier = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=tree_depth,
        min_samples_split=min_samples,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    return accuracy

storage_url = "postgresql://postgres:postgres@localhost:5432/postgres"
study_name = "breast_cancer_classification"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    direction="maximize",
    load_if_exists=False
)

study.optimize(objective, n_trials=50)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение:", study.best_value)

plot_optimization_history(study).show()
plot_slice(study).show()
plot_parallel_coordinate(study).show()
