from __future__ import annotations

import datetime
import json
import os
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def save_experiment(
    batch_size,
    min_number_images,
    balanced,
    bag_size,
    aggregation,
    dropout,
    class_weight,
    oversamplings,
    apply_augmentation,
    train_ratio,
    test_ratio,
    lr,
    epochs,
    counts,
    train_accuracy_list_tracker,
    test_accuracy_list_tracker,
    test_f1_tracker_list,
    test_roc_auc_tracker_list,
    test_pr_auc_tracker_list,
    test_roc_auc_all_mean,
    test_roc_auc_all_std,
    test_pr_auc_all_mean,
    test_pr_auc_all_std,
    fig_all_metrics,
    fig_all_together,
    seeds,
    cnf_matrices,
    fig_auc,
    img_dim,
    moderate_anaem,
    preds_record_list,
):

    data = {
        'batch_size': batch_size,
        'min_number_images': min_number_images,
        'balanced': balanced,
        'bag_size': bag_size,
        'aggregation': aggregation,
        'dropout': dropout,
        'class_weight': class_weight,
        'oversamplings': oversamplings,
        'apply_augmentation': apply_augmentation,
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'lr': lr,
        'epochs': epochs,
        'counts': counts,
        'train_accuracy_list_tracker': train_accuracy_list_tracker,
        'test_accuracy_list_tracker': test_accuracy_list_tracker,
        'test_f1_tracker_list': test_f1_tracker_list,
        'test_roc_auc_tracker_list': test_roc_auc_tracker_list,
        'test_pr_auc_tracker_list': test_pr_auc_tracker_list,
        'test_roc_auc_all_mean': test_roc_auc_all_mean,
        'test_roc_auc_all_std': test_roc_auc_all_std,
        'test_pr_auc_all_mean': test_pr_auc_all_mean,
        'test_pr_auc_all_std': test_pr_auc_all_std,
        'seeds': seeds,
        'cnf_matrices': cnf_matrices,
        'img_dim': img_dim,
        'moderate_anaem': moderate_anaem,
    }

    # convert to JSON format string
    data_str = json.dumps(data)

    # create DataFrame from this
    df = pd.DataFrame([{'Data': data_str}])

    # get the current date and time
    now = datetime.datetime.now()

    # convert to a string in your desired format
    timestamp = now.strftime('%Y%m%d_%H%M%S')

    # create the main directory and subdirectory with the current date and time
    main_directory = 'logs'
    sub_directory = os.path.join(main_directory, timestamp)
    os.makedirs(sub_directory, exist_ok=True)

    # create filename for csv
    filename = os.path.join(sub_directory, 'experiment.csv')

    # save to csv
    df.to_csv(filename, index=False)

    # Save your plots
    fig_all_metrics.savefig(
        os.path.join(
            sub_directory, 'separate_metrics.png',
        ), format='png',
    )
    fig_all_together.savefig(
        os.path.join(
            sub_directory, 'together_metrics.png',
        ), format='png',
    )
    fig_auc.savefig(
        os.path.join(
            sub_directory, 'auc_summary.png',
        ), format='png',
    )

    # Flatten list of lists of dictionaries into list of dictionaries
    flat_data_records = list(chain.from_iterable(preds_record_list))

    # Create DataFrame
    df = pd.DataFrame(flat_data_records)

    # get unique seed values
    unique_seeds = df['seed'].unique()

    # for each unique seed, create a separate DataFrame and save as CSV
    for seed in unique_seeds:
        df_seed = df[df['seed'] == seed]
        df_seed.to_csv(
            os.path.join(
                sub_directory, 'predictions_record_' + str(seed) + '.csv',
            ), index=False,
        )


def get_auc_summary(test_roc_auc_tracker_list, test_pr_auc_tracker_list, seeds):
    """
    This function calculates and prints the mean and standard deviation of the last ROC AUC (Receiver Operating Characteristic Area Under the Curve)
    and PR AUC (Precision-Recall Area Under the Curve) values for all runs (experiments), and returns these statistical metrics.

    Parameters:
    test_roc_auc_tracker_list (list): A list of lists, where each sublist contains the ROC AUC values of one run.
    test_pr_auc_tracker_list (list): A list of lists, where each sublist contains the PR AUC values of one run.
    seeds (list): A list of seeds used for the runs. The length of this list gives the number of runs.

    Returns:
    tuple: Mean and standard deviation of the last ROC AUC and PR AUC values for all runs.
    """

    test_last_roc_auc = []
    test_last_pr_auc = []

    # Calculate the last ROC AUC values for all runs
    for i in range(len(seeds)):
        test_last_roc_auc.append(test_roc_auc_tracker_list[i][-1])
    # Calculate the mean and standard deviation of the last ROC AUC values for all runs
    test_roc_auc_all_mean = np.mean(test_last_roc_auc)
    test_roc_auc_all_std = np.std(test_last_roc_auc)

    print('Mean ROC AUC: ', test_roc_auc_all_mean)
    print('Std ROC AUC : ', test_roc_auc_all_std)

    # Calculate the last PR AUC values for all runs
    for i in range(len(seeds)):
        test_last_pr_auc.append(test_pr_auc_tracker_list[i][-1])
    # Calculate the mean and standard deviation of the last PR AUC values for all runs
    test_pr_auc_all_mean = np.mean(test_last_pr_auc)
    test_pr_auc_all_std = np.std(test_last_pr_auc)

    print('Mean PR AUC : ', test_pr_auc_all_mean)
    print('Std PR AUC  : ', test_pr_auc_all_std)

    # Labels for the bars
    labels = ['ROC AUC', 'PR AUC']

    # Values for the bars
    means = [test_roc_auc_all_mean, test_pr_auc_all_mean]
    std_devs = [test_roc_auc_all_std, test_pr_auc_all_std]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Plot bars for mean values
    ax.bar(
        labels, means, yerr=std_devs, align='center',
        alpha=0.5, ecolor='black', capsize=10,
    )
    ax.set_ylabel('AUC Value')
    ax.set_title('Mean and Std Dev of ROC AUC and PR AUC')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.show()

    return test_roc_auc_all_mean, test_roc_auc_all_std, test_pr_auc_all_mean, test_pr_auc_all_std, fig


def calc_sens_spec_acc(df, positive_label, negative_label, col='true label'):
    TP = len(
        df[(df[col] == positive_label) & (
            df['predicted_label'] == positive_label
        )],
    )
    FN = len(
        df[(df[col] == positive_label) & (
            df['predicted_label'] == negative_label
        )],
    )
    TN = len(
        df[(df[col] == negative_label) & (
            df['predicted_label'] == negative_label
        )],
    )
    FP = len(
        df[(df[col] == negative_label) & (
            df['predicted_label'] == positive_label
        )],
    )

    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0

    return sensitivity, specificity, accuracy


def aggregate_summary(directory):

    dfs = []

    # 1. Read all the CSV files starting with 'predictions_record'
    for filename in os.listdir(directory):
        if filename.startswith('predictions_record_copy') and filename.endswith('.csv'):
            path = os.path.join(directory, filename)
            df = pd.read_csv(path)
            dfs.append(df)

    # 2. Combine all dataframes using concat
    combined_df = pd.concat(dfs, ignore_index=True)

    print('For classes:')
    # 3. Calculate sensitivity, specificity, and accuracy for each unique 'true label'
    sens_list, spec_list, acc_list = [], [], []

    for seed in combined_df['seed'].unique():
        subset = combined_df[combined_df['seed'] == seed]
        for seed in combined_df['seed'].unique():
            subset = combined_df[combined_df['seed'] == seed]
            if 1 in subset['true label'].unique():
                sensitivity, specificity, accuracy = calc_sens_spec_acc(
                    subset, 1, 0,
                )
                sens_list.append(sensitivity)
                spec_list.append(specificity)
                acc_list.append(accuracy)

    print(
        f'Sensitivity: Mean = {np.mean(sens_list):.2f}, Std = {np.std(sens_list):.2f}',
    )
    print(
        f'Specificity: Mean = {np.mean(spec_list):.2f}, Std = {np.std(spec_list):.2f}',
    )
    print(
        f'Accuracy: Mean = {np.mean(acc_list):.2f}, Std = {np.std(acc_list):.2f}',
    )

    plot_roc_curve(
        combined_df['true label'],
        combined_df['predicted probability'], 'ROC curve',
    )

    # PR Curve
    plot_pr_curve(
        combined_df['true label'],
        combined_df['predicted probability'], 'Precision-Recall curve',
    )

    # After calculating the metrics for classes:
    plot_boxplot(
        [acc_list, sens_list, spec_list], [
            'Accuracy', 'Sensitivity', 'Specificity',
        ], directory,
    )

    print()
    print('For subclasses:')
    # 4. Calculate sensitivity, specificity, and accuracy for each unique 'Diagnosis'
    sens_sub_list, spec_sub_list, acc_sub_list = [], [], []

    for seed in combined_df['seed'].unique():
        subset = combined_df[combined_df['seed'] == seed]
        for diagnosis in subset['Diagnosis'].unique():
            if diagnosis != 'Severe Malaria Anaemia':
                sub_subset = subset[(subset['Diagnosis'] == diagnosis) | (
                    subset['Diagnosis'] == 'Severe Malaria Anaemia'
                )]
                sensitivity, specificity, accuracy = calc_sens_spec_acc(
                    sub_subset, 1, 0, col='true label',
                )
                sens_sub_list.append(sensitivity)
                spec_sub_list.append(specificity)
                acc_sub_list.append(accuracy)

    print(
        f'Sensitivity: Mean = {np.mean(sens_sub_list):.2f}, Std = {np.std(sens_sub_list):.2f}',
    )
    print(
        f'Specificity: Mean = {np.mean(spec_sub_list):.2f}, Std = {np.std(spec_sub_list):.2f}',
    )
    print(
        f'Accuracy: Mean = {np.mean(acc_sub_list):.2f}, Std = {np.std(acc_sub_list):.2f}',
    )

    # Initialize list to store all TPRs for all ROC curves and base FPR for interpolation
    tprs, base_fpr = [], np.linspace(0, 1, 1000)

    for seed in combined_df['seed'].unique():
        subset = combined_df[combined_df['seed'] == seed]
        fpr, tpr, _ = roc_curve(
            subset['true label'], subset['predicted probability'], drop_intermediate=False,
        )

        # Define a set of desired FPR points at which to interpolate
        # You can adjust the number of points as needed
        desired_fpr = np.linspace(0, 1, 1000)

        # Interpolate TPR values to match the desired FPR points
        # interp_tpr = np.interp(desired_fpr, fpr, tpr)
        interp_linear_spline = interp1d(
            fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1),
        )
        fpr = desired_fpr
        # tpr=interp_tpr
        interp_tpr_linear_spline = interp_linear_spline(desired_fpr)
        tprs.append((desired_fpr, interp_tpr_linear_spline))
        # tprs.append((fpr, tpr))

    # Get mean and std of ROC curves
    mean_tpr, std_tpr, mean_auc, std_auc = get_mean_and_std_roc(tprs, base_fpr)

    # Compute the area under the mean ROC curve
    mean_auc = auc(base_fpr, mean_tpr)

    plot_mean_roc_curve(base_fpr, mean_tpr, std_tpr, mean_auc, std_auc)

    # Initialize list to store all Precisions for all PR curves and base Recall for interpolation
    precisions, base_recall = [], np.linspace(0, 1, 100)

    for seed in combined_df['seed'].unique():
        subset = combined_df[combined_df['seed'] == seed]
        precision, recall, _ = precision_recall_curve(
            subset['true label'], subset['predicted probability'],
        )
        precisions.append((recall, precision))

    # Get mean and std of PR curves
    mean_precision, std_precision, mean_ap, std_ap = get_mean_and_std_pr(
        precisions, base_recall,
    )

    # Compute the average precision for the mean PR curve
    mean_avg_precision = average_precision_score(
        combined_df['true label'], combined_df['predicted probability'],
    )

    plot_mean_pr_curve(
        base_recall, mean_precision,
        std_precision, mean_ap, std_ap,
    )


# Plotting function
def plot_roc_curve(y_true, y_pred_prob, title):
    fpr, tpr, thresholds = roc_curve(
        y_true, y_pred_prob, drop_intermediate=False,
    )
    # Define a set of desired FPR points at which to interpolate
    # You can adjust the number of points as needed
    desired_fpr = np.linspace(0, 1, 1000)

    # Interpolate TPR values to match the desired FPR points
    interp_tpr = np.interp(desired_fpr, fpr, tpr)
    fpr = desired_fpr
    tpr = interp_tpr
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (area = {roc_auc:.2f})',
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


def plot_boxplot(data, labels, directory_name):
    """
    Plot a boxplot for the data.

    Parameters:
    data (list of lists): List containing lists of values to be plotted.
    labels (list): List of labels for the data.
    directory_name (str): Name of the directory, which will be used as the plot title.
    """
    fig, ax = plt.subplots(figsize=(4, 6))  # Adjust figure size here
    # Set the widths parameter to adjust the width of the boxes
    ax.boxplot(data, widths=0.6)
    ax.set_xticklabels(labels)
    ax.set_title(f'{directory_name[-15:]}')
    plt.tight_layout()  # Adjust layout so everything fits nicely
    plt.show()


# New PR Curve function
def plot_pr_curve(y_true, y_pred_prob, title):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)

    plt.figure()
    plt.plot(
        recall, precision, color='blue', lw=2,
        label=f'PR curve (area = {pr_auc:.2f})',
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


def get_mean_and_std_roc(tprs, base_fpr):
    """
    Get mean and standard deviation of interpolated TPRs and AUCs.

    Parameters:
    - tprs (list): List of True Positive Rates.
    - base_fpr (numpy array): Common False Positive Rates.

    Returns:
    - mean_tpr (numpy array): Mean True Positive Rate.
    - std_tpr (numpy array): Standard Deviation of True Positive Rates.
    - mean_auc (float): Mean AUC.
    - std_auc (float): Standard Deviation of AUCs.
    """

    tpr_array = np.array([
        interp1d(fpr, tpr, bounds_error=True, fill_value=0.)(
            base_fpr,
        ) for fpr, tpr in tprs
    ])
    auc_values = [auc(base_fpr, tpr) for tpr in tpr_array]

    mean_tpr = np.mean(tpr_array, axis=0)
    std_tpr = np.std(tpr_array, axis=0)
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)

    return mean_tpr, std_tpr, mean_auc, std_auc


def plot_mean_roc_curve(base_fpr, mean_tpr, std_tpr, mean_auc, std_auc):
    """
    Plot mean ROC curve with envelopes for the standard deviation.

    Parameters:
    - base_fpr (numpy array): Common False Positive Rates.
    - mean_tpr (numpy array): Mean True Positive Rate.
    - std_tpr (numpy array): Standard Deviation of True Positive Rates.
    - mean_auc (float): Area under the mean ROC curve.
    - std_auc (float): Standard Deviation of AUC values.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(
        base_fpr, mean_tpr, color='b',
        label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
    )
    plt.fill_between(
        base_fpr, mean_tpr - std_tpr, mean_tpr +
        std_tpr, color='grey', alpha=0.3, label='± 1 std. dev.',
    )
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve with Envelopes')
    plt.legend(loc='lower right')
    plt.show()


def get_mean_and_std_pr(precisions, base_recall):
    """
    Get mean and standard deviation of interpolated Precisions and AP values.

    Parameters:
    - precisions (list): List of Precisions.
    - base_recall (numpy array): Common Recall values.

    Returns:
    - mean_precision (numpy array): Mean Precision.
    - std_precision (numpy array): Standard Deviation of Precisions.
    - mean_ap (float): Mean Average Precision.
    - std_ap (float): Standard Deviation of AP values.
    """
    precision_array = np.array([
        interp1d(recall, precision, bounds_error=False, fill_value=0.)(
            base_recall,
        ) for recall, precision in precisions
    ])
    ap_values = [auc(base_recall, precision) for precision in precision_array]

    mean_precision = np.mean(precision_array, axis=0)
    std_precision = np.std(precision_array, axis=0)
    mean_ap = np.mean(ap_values)
    std_ap = np.std(ap_values)

    return mean_precision, std_precision, mean_ap, std_ap


def plot_mean_pr_curve(base_recall, mean_precision, std_precision, mean_ap, std_ap):
    """
    Plot mean Precision-Recall curve with envelopes for the standard deviation.

    Parameters:
    - base_recall (numpy array): Common Recall values.
    - mean_precision (numpy array): Mean Precision.
    - std_precision (numpy array): Standard Deviation of Precisions.
    - mean_ap (float): Mean Average Precision.
    - std_ap (float): Standard Deviation of AP values.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(
        base_recall, mean_precision, color='b',
        label=f'Mean PR (AP = {mean_ap:.2f} ± {std_ap:.2f})',
    )
    plt.fill_between(
        base_recall, mean_precision - std_precision, mean_precision +
        std_precision, color='grey', alpha=0.3, label='± 1 std. dev.',
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Mean Precision-Recall Curve with Envelopes')
    plt.legend(loc='lower left')
    plt.show()
