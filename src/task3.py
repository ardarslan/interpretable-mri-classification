import os
import pprint
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from typing import Dict, Any

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_radiomics_dataset


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    # TODO: Add task-specific CLI arguments here.
    return parser


def build_log_file_path(cfg: Dict[str, Any]) -> str:
    """Construct path to log file from configuration."""
    # TODO: Customize this if needed.
    path = os.path.join(
        cfg["checkpoints"],
        "task3",
    )
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"experiment-{cfg['id']}.log")
    return file_path


def train_and_eval_interpretable_model(X_train, X_valid, X_test, y_train, y_valid, y_test, logger):

    ## step 1: hyperparameter search on validation data
    Cs = np.logspace(start=-4,stop=4,num=9)

    acc_scores = {}
    nonzero_coefs = {}

    for temp_c in Cs:
        scaler = StandardScaler()
        model = LogisticRegression(penalty='l1',solver='saga',C=temp_c, random_state=0)

        X_train_ = scaler.fit_transform(X_train)
        X_valid_ = scaler.transform(X_valid)

        model.fit(X_train_,y_train)
        valid_preds = model.predict(X_valid_)

        acc_scores[temp_c] = accuracy_score(y_true=y_valid,y_pred=valid_preds)
        nonzero_coefs[temp_c] = int(np.sum(model.coef_ != 0.0))

    ## step 2: plot accuracy and non-zero coefficients against different Cs
    fig, ax1 = plt.subplots(figsize=(8,4))

    line2, = plt.plot(list(acc_scores.keys()),list(acc_scores.values()),alpha=0.7,label='accuracy',marker='*')
    plt.xscale('log')
    plt.xlabel('C (log-scaled)')
    ax1.set_ylabel('accuracy')

    ax2 = ax1.twinx()
    line3, = plt.plot(list(nonzero_coefs.keys()),list(nonzero_coefs.values()),alpha=0.7,color='red',label='non-zero coef count',marker='*')

    fig.tight_layout()  
    plt.legend(handles=[line2,line3])
    plt.title('Regularization coefficient (C) optimization for validation data')
    plt.xticks(Cs)
    plt.grid(alpha=0.2,axis='both')
    ax2.set_ylabel('coefficient count')
    plt.tight_layout()
    plt.savefig('../plots/hpo')

    ## step 3: train model with selected C and merge train and validation data

    X_train_valid = pd.concat([X_train,X_valid])
    y_train_valid = np.append(y_train,y_valid)

    scaler = StandardScaler()
    # C=1e0 is selected from validation performance
    model = LogisticRegression(penalty='l1',solver='saga',C=1e0, random_state=0)

    X_train_valid_ = scaler.fit_transform(X_train_valid)
    X_test_ = scaler.transform(X_test)

    model.fit(X_train_valid_,y_train_valid)
    test_preds = model.predict(X_test_)

    test_acc = accuracy_score(y_true=y_test,y_pred=test_preds)
    non_zero_coef_count = int(np.sum(model.coef_ != 0.0))
    logger.info(f'test accuracy: {test_acc}, non_zero_coef_count: {non_zero_coef_count}')

    coef_magnitude_df = pd.DataFrame({'col1':np.abs(model.coef_[0]),'col2':X_train.keys()}).sort_values(by='col1')
    coef_magnitude_df = coef_magnitude_df.tail(20)

    plt.figure(figsize=(6,8))
    plt.xlabel('magnitude')
    plt.title('LogReg.L1-Reg Feat Importance, C=1e0')
    plt.grid(alpha=0.3,axis='x')
    plt.barh(coef_magnitude_df['col2'],coef_magnitude_df['col1'])
    plt.savefig('../plots/logreg_implicit_feat_imp_c1e0',bbox_inches='tight')
    
    return


def get_permutation_feature_importances(X_train, X_valid, X_test, y_train, y_valid, y_test, logger):

    X_train_valid = pd.concat([X_train,X_valid])
    y_train_valid = np.append(y_train,y_valid)

    scaler = StandardScaler()
    model = LogisticRegression(penalty='elasticnet',solver='saga',C=1e0, random_state=0,l1_ratio = 0.5)

    X_train_valid_ = scaler.fit_transform(X_train_valid)
    X_test_ = scaler.transform(X_test)

    model.fit(X_train_valid_,y_train_valid)

    test_preds = model.predict(X_test_)

    test_acc = accuracy_score(y_true=y_test,y_pred=test_preds)
    logger.info(f'test_acc for permutation feat importance: {test_acc}')

    # TEST DATA
    result_test = permutation_importance(
        model, X_test_, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )

    model_importances = pd.Series(result_test.importances_mean, index=list(X_train.keys()))
    model_importances = model_importances[model_importances != 0.0]
    model_importances = model_importances.sort_values(ascending=False)

    # for better visual, eleminate some unimportant features
    model_importances = model_importances[(model_importances>0) | (model_importances<-0.01)]

    fig, ax = plt.subplots(figsize=(12,7))
    model_importances.plot.bar()
    ax.set_title("Permutation Importance on test data for ElasticNet Logistic Regression")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.grid(alpha=0.5,axis='both')
    plt.savefig('../plots/permutation_feat_imp_for_test',bbox_inches='tight')


    # TRAIN DATA
    result_train = permutation_importance(
        model, X_train_valid_, y_train_valid, n_repeats=5, random_state=42, n_jobs=-1
    )

    model_importances = pd.Series(result_train.importances_mean, index=list(X_train.keys()))
    model_importances = model_importances[model_importances != 0.0]
    model_importances = model_importances.sort_values(ascending=False)

    # for better visual, eleminate some unimportant features
    model_importances = model_importances[np.abs(model_importances) > 0.005]
    model_importances = model_importances[(model_importances>0) | (model_importances<-0.005)]

    fig, ax = plt.subplots(figsize=(12,7))
    model_importances.plot.bar()
    ax.set_title("Permutation Importance on train data for ElasticNet Logistic Regression")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.grid(alpha=0.5,axis='both')
    plt.savefig('../plots/permutation_feat_imp_for_train',bbox_inches='tight')

    return


def get_partial_dependence_plots(X_train, X_valid, X_test, y_train, y_valid, y_test, logger, feature_names):

    X_train_valid = pd.concat([X_train,X_valid])
    y_train_valid = np.append(y_train,y_valid)

    scaler = StandardScaler()
    model = LogisticRegression(penalty='elasticnet',solver='saga',C=1e0, random_state=0,l1_ratio = 0.5)

    X_train_valid_ = scaler.fit_transform(X_train_valid)
    X_test_ = scaler.transform(X_test)

    X_train_valid_ = pd.DataFrame(data=X_train_valid_,columns=X_train.keys().values)
    X_test_ = pd.DataFrame(data=X_test_,columns=X_train.keys().values)

    model.fit(X_train_valid_,y_train_valid)

    test_preds = model.predict(X_test_)

    test_acc = accuracy_score(y_true=y_test,y_pred=test_preds)
    logger.info(f'test_acc for permutation feat importance: {test_acc}')

    display = PartialDependenceDisplay.from_estimator(
        model,
        X_train_valid_,
        features=feature_names,
        random_state=0,
        grid_resolution=20,
    )

    display.figure_.subplots_adjust(right=2.0,top=2.5)
    plt.savefig('../plots/partial_dependence_plots',bbox_inches='tight')

    return


def main() -> None:
    arg_parser = init_argument_parser()
    arg_parser = extend_argument_parser(arg_parser)

    cfg = arg_parser.parse_args().__dict__

    set_seeds(cfg["seed"])
    if cfg["id"] is None:
        init_id(cfg)

    logger = init_logger(path_to_log=build_log_file_path(cfg))
    logger.info(pprint.pformat(cfg, indent=4))

    logger.info("Loading radiomics dataset...")
    data_dir = os.path.join(cfg["data"], "radiomics")
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_radiomics_dataset(
        data_dir=data_dir
    )
    logger.info("Radiomics dataset loaded!")
    
    # INTERPRETABLE MODEL: Logistic Regression with L1 Regularization
    train_and_eval_interpretable_model(X_train, X_valid, X_test, y_train, y_valid, y_test, logger)
   
    # POST-HOC METHOD 1: Permutation Feature Importance
    get_permutation_feature_importances(X_train, X_valid, X_test, y_train, y_valid, y_test, logger)

    # POST-HOC METHOD 2: Partial Dependence Plots
    random_feature_subset = ['wavelet-LH_gldm_GrayLevelVariance', 'original_shape2D_Sphericity',
        'original_firstorder_10Percentile',
        'original_firstorder_90Percentile','wavelet-LH_firstorder_Uniformity',
        'wavelet-LH_firstorder_Variance',
        'wavelet-LH_glcm_Autocorrelation',
        'wavelet-LH_glcm_ClusterProminence',
        'wavelet-LH_glcm_ClusterShade']

    get_partial_dependence_plots(X_train, X_valid, X_test, y_train, y_valid, y_test, logger, random_feature_subset)



    return


if "__main__" == __name__:
    main()
