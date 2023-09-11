import pandas as pd
import datetime
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def save_experiment(batch_size,
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
        preds_record_list
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
        'cnf_matrices':cnf_matrices,
        'img_dim': img_dim,
        'moderate_anaem':moderate_anaem,
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
    main_directory = 'Experiments_log'
    sub_directory = os.path.join(main_directory, timestamp)
    os.makedirs(sub_directory, exist_ok=True)

    # create filename for csv
    filename = os.path.join(sub_directory, 'experiment.csv')
    
    # save to csv
    df.to_csv(filename, index=False)

    # Save your plots
    fig_all_metrics.savefig(os.path.join(sub_directory, 'separate_metrics.png'), format='png')
    fig_all_together.savefig(os.path.join(sub_directory, 'together_metrics.png'), format='png')
    fig_auc.savefig(os.path.join(sub_directory, 'auc_summary.png'), format='png')


    # Flatten list of lists of dictionaries into list of dictionaries
    flat_data_records = list(chain.from_iterable(preds_record_list))

    # Create DataFrame
    df = pd.DataFrame(flat_data_records)

    # get unique seed values
    unique_seeds = df['seed'].unique()

    # for each unique seed, create a separate DataFrame and save as CSV
    for seed in unique_seeds:
        df_seed = df[df['seed'] == seed]
        df_seed.to_csv(os.path.join(sub_directory,'predictions_record_' + str(seed) + '.csv'), index=False)


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

    test_last_roc_auc=[]
    test_last_pr_auc=[]

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
    ax.bar(labels, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('AUC Value')
    ax.set_title('Mean and Std Dev of ROC AUC and PR AUC')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.show()

    return test_roc_auc_all_mean, test_roc_auc_all_std, test_pr_auc_all_mean, test_pr_auc_all_std, fig

def calc_sens_spec_acc(df, positive_label, negative_label, col='true label'):
    TP = len(df[(df[col] == positive_label) & (df['predicted_label'] == positive_label)])
    FN = len(df[(df[col] == positive_label) & (df['predicted_label'] == negative_label)])
    TN = len(df[(df[col] == negative_label) & (df['predicted_label'] == negative_label)])
    FP = len(df[(df[col] == negative_label) & (df['predicted_label'] == positive_label)])
    
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
    
    return sensitivity, specificity, accuracy

def aggregate_summary(directory):

    dfs = []

    # 1. Read all the CSV files starting with 'predictions_record'
    for filename in os.listdir(directory):
        if filename.startswith('predictions_record') and filename.endswith('.csv'):
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
                sensitivity, specificity, accuracy = calc_sens_spec_acc(subset, 1, 0)
                sens_list.append(sensitivity)
                spec_list.append(specificity)
                acc_list.append(accuracy)

    print(f"Sensitivity: Mean = {np.mean(sens_list):.2f}, Std = {np.std(sens_list):.2f}")
    print(f"Specificity: Mean = {np.mean(spec_list):.2f}, Std = {np.std(spec_list):.2f}")
    print(f"Accuracy: Mean = {np.mean(acc_list):.2f}, Std = {np.std(acc_list):.2f}")

    plot_roc_curve(combined_df['true label'], combined_df['predicted probability'], "ROC curve")

    # PR Curve
    plot_pr_curve(combined_df['true label'], combined_df['predicted probability'], "Precision-Recall curve")

    # After calculating the metrics for classes:
    plot_boxplot([acc_list, sens_list, spec_list], ['Accuracy', 'Sensitivity', 'Specificity'], directory)

    print()
    print('For subclasses:')
    # 4. Calculate sensitivity, specificity, and accuracy for each unique 'Diagnosis'
    sens_sub_list, spec_sub_list, acc_sub_list = [], [], []
    
    for seed in combined_df['seed'].unique():
        subset = combined_df[combined_df['seed'] == seed]
        for diagnosis in subset['Diagnosis'].unique():
            if diagnosis != 'Severe Malaria Anaemia':
                sub_subset = subset[(subset['Diagnosis'] == diagnosis) | (subset['Diagnosis'] == 'Severe Malaria Anaemia')]
                sensitivity, specificity, accuracy = calc_sens_spec_acc(sub_subset, 1, 0, col='true label')
                sens_sub_list.append(sensitivity)
                spec_sub_list.append(specificity)
                acc_sub_list.append(accuracy)

    print(f"Sensitivity: Mean = {np.mean(sens_sub_list):.2f}, Std = {np.std(sens_sub_list):.2f}")
    print(f"Specificity: Mean = {np.mean(spec_sub_list):.2f}, Std = {np.std(spec_sub_list):.2f}")
    print(f"Accuracy: Mean = {np.mean(acc_sub_list):.2f}, Std = {np.std(acc_sub_list):.2f}")

# Plotting function
def plot_roc_curve(y_true, y_pred_prob, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
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
    ax.boxplot(data, widths=0.6)  # Set the widths parameter to adjust the width of the boxes
    ax.set_xticklabels(labels)
    ax.set_title(f'{directory_name[-15:]}')
    plt.tight_layout()  # Adjust layout so everything fits nicely
    plt.show()


# New PR Curve function
def plot_pr_curve(y_true, y_pred_prob, title):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()

