import os
import pprint
from argparse import ArgumentParser
import pickle
from pathlib import Path

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from rulefit import RuleFit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_radiomics_dataset


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="Number of gradient boosted trees.",
    )
    return parser


def build_checkpoints_path(cfg: Dict[str, Any]) -> str:
    """Construct path to task 1 checkpoints folder from configuration."""
    return os.path.join(cfg["checkpoints"], "task3")


def build_log_file_path(cfg: Dict[str, Any]) -> str:
    """Construct path to log file from configuration."""
    path = build_checkpoints_path(cfg)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"experiment-{cfg['id']}.log")
    return file_path


def save_rulefit_classifier(rulefit: RuleFit, model_path: str) -> None:
    """Save RuleFit classifier to given path."""
    with open(model_path, "wb") as f:
        pickle.dump(obj=rulefit, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_rulefit_classifier(model_path: str) -> RuleFit:
    """Load RuleFit classifier from given path."""
    with open(model_path, "rb") as f:
        model = pickle.load(file=f)
    return model


class Task3RuleFit:
    """Task 3 RuleFit experiments."""

    def __init__(self) -> None:
        """Set up task 3 RuleFit experiments."""
        arg_parser = init_argument_parser()
        arg_parser = extend_argument_parser(arg_parser)

        self.cfg = arg_parser.parse_args().__dict__

        set_seeds(self.cfg["seed"])
        if self.cfg["id"] is None:
            init_id(self.cfg)

        self.logger = init_logger(path_to_log=build_log_file_path(self.cfg))
        self.logger.info(pprint.pformat(self.cfg, indent=4))

        model_dir = os.path.join(Path(self.cfg["data"]).parent, "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "rulefit.pickle")

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

    def train_rulefit_classifier(self) -> RuleFit:
        """Train RuleFit classifier based on given configuration."""
        rulefit = RuleFit(
            tree_generator=RandomForestClassifier(
                n_estimators=self.cfg["n_estimators"], random_state=self.cfg["seed"],
            ),
            rfmode="classify",
            random_state=self.cfg["seed"],
        )
        X_train_valid, y_train_valid = self.combine_train_valid()
        rulefit.fit(
            X_train_valid, y_train_valid, feature_names=self.X_train.columns,
        )
        save_rulefit_classifier(rulefit=rulefit, model_path=self.model_path)
        return rulefit

    def extract_rules(self, rulefit: RuleFit) -> pd.DataFrame:
        """Extract a sorted list of rules according to support."""
        rules = rulefit.get_rules()
        # Remove pure variables.
        rules = rules[rules["type"] != "linear"]
        # Remove insignificant rules.
        rules = rules[rules["coef"] != 0]
        return rules.sort_values("support", ascending=False)

    def evaluate_rulefit_classifier(self, rulefit: RuleFit) -> None:
        rules = self.extract_rules(rulefit)
        self.logger.info("Rules:\n%s", rules)
        rules_filename = os.path.join(
            build_checkpoints_path(self.cfg), f"rules-{self.cfg['id']}.csv"
        )
        rules.to_csv(rules_filename)
        self.logger.info("Saved rules as CSV to %s!", rules_filename)

        metrics = {
            "accuracy": accuracy_score,
            "balanced accuracy": balanced_accuracy_score,
        }

        for description, metric in metrics.items():
            self.logger.info(
                "Training set %s: %f",
                description,
                metric(self.y_train, rulefit.predict(self.X_train.values)),
            )
            self.logger.info(
                "Validation set %s: %f",
                description,
                metric(self.y_valid, rulefit.predict(self.X_valid.values)),
            )
            self.logger.info(
                "Test set %s: %f",
                description,
                metric(self.y_test, rulefit.predict(self.X_test.values)),
            )

    def run(self) -> None:
        """Run task 3 RuleFit experiments."""
        if self.cfg["retrain"]:
            rulefit = self.train_rulefit_classifier()
        else:
            rulefit = load_rulefit_classifier(self.model_path)

        self.evaluate_rulefit_classifier(rulefit)


if "__main__" == __name__:
    task3_rulefit = Task3RuleFit()
    task3_rulefit.run()
