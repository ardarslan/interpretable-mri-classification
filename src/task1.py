import os
import pprint
from argparse import ArgumentParser
import pickle

from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost.sklearn import XGBClassifier

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_radiomics_dataset


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="Number of gradient boosted trees.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Max. tree depth of base learners.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Boosting learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Min. loss to make further partition on leaf node.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.9,
        help="Subsample ratio of training instance.",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="If set, hyperparameter grid search for XGBoost classifier will"
        + " be performed.",
    )
    return parser


def build_checkpoints_path(cfg: Dict[str, Any]) -> str:
    """Construct path to task 1 checkpoints folder from configuration."""
    return os.path.join(cfg["checkpoints"], "task1")


def build_log_file_path(cfg: Dict[str, Any]) -> str:
    """Construct path to log file from configuration."""
    path = build_checkpoints_path(cfg)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"experiment-{cfg['id']}.log")
    return file_path


def select_important_features(
    features: List[str], feature_importances: List[float]
) -> Dict[str, float]:
    assert len(features) == len(
        feature_importances
    ), "Need as many feature importances as features!"
    important_features = zip(features, feature_importances)
    return {t[0]: t[1] for t in important_features if t[1] > 0.0}


def save_xgb_classifier(xgbc: XGBClassifier, model_path: str) -> None:
    """Save XGBoost classifier to given path."""
    with open(model_path, "wb") as f:
        pickle.dump(obj=xgbc, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_xgb_classifier(model_path: str) -> XGBClassifier:
    """Load XGBoost classifier from given path."""
    with open(model_path, "rb") as f:
        model = pickle.load(file=f)
    return model


class Task1:
    """Task 1 experiments."""

    def __init__(self) -> None:
        """Set up task 1 experiments."""
        arg_parser = init_argument_parser()
        arg_parser = extend_argument_parser(arg_parser)

        self.cfg = arg_parser.parse_args().__dict__

        set_seeds(self.cfg["seed"])
        if self.cfg["id"] is None:
            init_id(self.cfg)

        self.logger = init_logger(path_to_log=build_log_file_path(self.cfg))
        self.logger.info(pprint.pformat(self.cfg, indent=4))

        self.model_path = os.path.join(
            build_checkpoints_path(self.cfg), "xgbc.pickle"
        )

        self.load_dataset()

    def load_dataset(self) -> None:
        self.logger.info("Loading radiomics dataset...")
        data_dir = os.path.join(self.cfg["data"], "radiomics")
        (
            self.X_train,
            self.y_train,
            self.X_valid,
            self.y_valid,
            self.X_test,
            self.y_test,
        ) = get_radiomics_dataset(data_dir=data_dir)
        self.logger.info("Radiomics dataset loaded!")

        self.logger.info("X_train shape: %s", self.X_train.shape)
        self.logger.info("y_train shape: %s", self.y_train.shape)
        self.logger.info("X_valid shape: %s", self.X_valid.shape)
        self.logger.info("y_valid shape: %s", self.y_valid.shape)
        self.logger.info("X_test shape: %s", self.X_test.shape)
        self.logger.info("y_test shape: %s", self.y_test.shape)

    def combine_train_valid(self) -> Tuple[np.ndarray, np.ndarray]:
        X_train_valid = np.concatenate(
            (self.X_train.to_numpy(), self.X_valid.to_numpy())
        )
        y_train_valid = np.concatenate((self.y_train, self.y_valid))
        return X_train_valid, y_train_valid

    def search_grid(self) -> GridSearchCV:
        """Perform grid search over XGBoost hyperparamters, return result."""
        parameter_grid = {
            "n_estimators": [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
            ],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1],
            "gamma": [0.0, 0.25, 0.5, 0.75, 1.0],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        }

        X_train_valid, y_train_valid = self.combine_train_valid()
        self.logger.info("X_train_valid shape: %s", self.X_train.shape)
        self.logger.info("y_train_valid shape: %s", self.y_train.shape)

        xgbc = XGBClassifier(random_state=self.cfg["seed"])

        grid_search = GridSearchCV(
            xgbc,
            parameter_grid,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
            cv=5,
            verbose=4,
            return_train_score=True,
        )

        grid_search.fit(X_train_valid, y_train_valid)
        return grid_search

    def train_xgb_classifier(self) -> XGBClassifier:
        """Train XGBoost classifier based on given configuration."""
        if self.cfg["grid_search"]:

            self.logger.info("Starting grid search for XGBoost Classifier...")
            grid_search = self.search_grid()
            self.logger.info("Grid search complete!")

            xgbc = grid_search.best_estimator_
            save_xgb_classifier(xgbc=xgbc, model_path=self.model_path)

            self.logger.info(
                "Best hyperparameter settings: %s",
                pprint.pformat(grid_search.best_params_, indent=4),
            )
        else:
            xgbc = XGBClassifier(
                n_estimators=self.cfg["n_estimators"],
                max_depth=self.cfg["max_depth"],
                learning_rate=self.cfg["lr"],
                gamma=self.cfg["gamma"],
                subsample=self.cfg["subsample"],
            )
            X_train_valid, y_train_valid = self.combine_train_valid()
            xgbc.fit(X_train_valid, y_train_valid)
            save_xgb_classifier(xgbc=xgbc, model_path=self.model_path)
        return xgbc

    def evaluate_xgb_classifier(self, xgbc: XGBClassifier) -> None:
        important_features = select_important_features(
            self.X_train.columns.to_list(), xgbc.feature_importances_.tolist()
        )

        self.logger.info(
            "Important features (%d/%d): %s",
            len(important_features),
            len(self.X_train.columns),
            pprint.pformat(important_features, indent=4),
        )

        metrics = {
            "accuracy": accuracy_score,
            "balanced accuracy": balanced_accuracy_score,
        }

        for description, metric in metrics.items():
            self.logger.info(
                "Training set %s: %f",
                description,
                metric(self.y_train, xgbc.predict(self.X_train)),
            )
            self.logger.info(
                "Validation set %s: %f",
                description,
                metric(self.y_valid, xgbc.predict(self.X_valid)),
            )
            self.logger.info(
                "Test set %s: %f",
                description,
                metric(self.y_test, xgbc.predict(self.X_test)),
            )

    def run(self) -> None:
        """Run task 1."""
        if self.cfg["retrain"]:
            xgbc = self.train_xgb_classifier()
        else:
            xgbc = load_xgb_classifier(self.model_path)

        self.evaluate_xgb_classifier(xgbc)


if "__main__" == __name__:
    task1 = Task1()
    task1.run()
