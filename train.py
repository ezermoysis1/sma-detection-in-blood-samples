from __future__ import annotations

import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.train.Experiment import Experiment
from src.train.plotting import plot_metrics_separately
from src.train.plotting import plot_metrics_together
from src.train.utils import get_auc_summary
from src.train.utils import save_experiment


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig) -> None:
    base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    base_dir = base_dir.replace('/outputs', '')

    # Combine with the relative directory
    new_path = os.path.join(base_dir, cfg.paths.data_dir)

    seeds = [42, 105, 4, 21]  # 42, 105, 4, 21
    exp = []
    train_accuracy_list_tracker = []
    test_accuracy_list_tracker = []
    test_f1_tracker_list = []
    test_roc_auc_tracker_list = []
    test_pr_auc_tracker_list = []
    cnf_matrix_list = []
    plots = []
    diagn_distr_list = []
    diagnosis_pred_distr_list = []
    preds_record_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('This will run on', device)

    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        torch.manual_seed(seeds[i])
        random.seed(seeds[i])

        exp = Experiment(
            seeds[i],
            cfg.img_dim,
            new_path,
            cfg.min_number_images,
            cfg.train_ratio,
            cfg.batch_size,
            cfg.apply_augmentation,
            oversampling=cfg.oversamplings,
            class_weight=cfg.class_weight,
            dropout=cfg.dropout,
            aggregation=cfg.aggregation,
            lr=cfg.lr,
        )

        counts = exp.get_class_counts()

        diagn_distr_list.append(exp.get_loader_diagnosis_distribution())

        exp.train(
            epochs=cfg.epochs, bag_size=cfg.bag_size,
            class_weight=cfg.class_weight,
        )

        train_accuracy_list, test_accuracy_list, test_f1_tracker, test_roc_auc_tracker, test_pr_auc_tracker, cnf_matrix, diagnosis_pred_distr, preds_record = exp.return_metrics()

        exp.plot_metrics()
        exp.plot_cnf()

        train_accuracy_list_tracker.append(train_accuracy_list)
        test_accuracy_list_tracker.append(test_accuracy_list)
        test_f1_tracker_list.append(test_f1_tracker)
        test_roc_auc_tracker_list.append(test_roc_auc_tracker)
        test_pr_auc_tracker_list.append(test_pr_auc_tracker)
        cnf_matrix_list.append(cnf_matrix)
        diagnosis_pred_distr_list.append(diagnosis_pred_distr)
        preds_record_list.append(preds_record)

    # Metric figures and confusion matrix
    fig_all_metrics = plot_metrics_separately(
        train_accuracy_list_tracker, test_accuracy_list_tracker, test_f1_tracker_list, test_roc_auc_tracker_list,
        test_pr_auc_tracker_list, cnf_matrix_list=cnf_matrix_list, diagn_dist_list=diagn_distr_list, diagnosis_pred_distr_list=diagnosis_pred_distr_list,
    )
    fig_all_together = plot_metrics_together(
        train_accuracy_list_tracker, test_accuracy_list_tracker, test_f1_tracker_list,
    )

    # Handling AUCs:
    test_roc_auc_all_mean, test_roc_auc_all_std, test_pr_auc_all_mean, test_pr_auc_all_std, fig = get_auc_summary(
        test_roc_auc_tracker_list, test_pr_auc_tracker_list, seeds,
    )

    # Log the Experiment inputs and outputs
    cnf_matrices = [array.tolist() for array in cnf_matrix_list]

    save_experiment(
        cfg, counts, train_accuracy_list_tracker, test_accuracy_list_tracker, test_f1_tracker_list,
        test_roc_auc_tracker_list, test_pr_auc_tracker_list, test_roc_auc_all_mean, test_roc_auc_all_std,
        test_pr_auc_all_mean, test_pr_auc_all_std, fig_all_metrics, fig_all_together, seeds, cnf_matrices,
        fig, preds_record_list,
    )


if __name__ == '__main__':
    main()
