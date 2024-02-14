from __future__ import annotations

import os
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metrics_separately(
    train_accuracy_list_tracker, test_accuracy_list_tracker,
    test_f1_tracker_list, test_roc_auc_tracker_list,
    test_pr_auc_tracker_list, cnf_matrix_list, diagn_dist_list, diagnosis_pred_distr_list,
):
    """
    Plots various metrics separately for each epoch iteration.

    Args:
        train_accuracy_list_tracker (list of lists): List of training accuracy values for each epoch.
        test_accuracy_list_tracker (list of lists): List of test accuracy values for each epoch.
        test_f1_tracker_list (list of lists): List of test F1 score values for each epoch.
        test_roc_auc_tracker_list (list of lists): List of test ROC AUC values for each epoch.
        test_pr_auc_tracker_list (list of lists): List of test PR AUC values for each epoch.
        cnf_matrix_list (list of arrays): List of confusion matrices for each epoch.
        diagn_dist_list (list of dicts): List of diagnosis distributions for training and test data for each epoch.
        diagnosis_pred_distr_list (list of dicts): List of diagnosis distributions for training and test data, including correct predictions, for each epoch.

    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure.
    """

    fig, axes = plt.subplots(
        4, len(train_accuracy_list_tracker), figsize=(
            len(train_accuracy_list_tracker)*5, 25,
        ),
    )
    counter = 0

    # Update color dictionary
    color_dict = {

        'Severe Malaria Anaemia': 'red',
        'Malaria, Severe Anaemia, No SMA': 'orange',
        'No Malaria, Anaemia': 'yellow',
        'No Malaria, Severe Anaemia': 'pink',
        'Malaria, No Anaemia': 'blue',
        'Malaria, Anaemia, No Severe': 'magenta',
        'No Malaria, No Anaemia': 'green',
        'Unclassified': 'cyan',
    }

    legend_patches = [
        mpatches.Patch(
            color=color, label=diagnosis,
        ) for diagnosis, color in color_dict.items()
    ]

    for i in range(len(train_accuracy_list_tracker)):
        axes[0][i].plot(
            train_accuracy_list_tracker[counter],
            color='red', label='Training accuracy (%)',
        )
        axes[0][i].plot(
            test_accuracy_list_tracker[counter],
            color='blue', label='Test accuracy (%)',
        )
        axes[0][i].plot(
            test_f1_tracker_list[counter],
            color='green', label='Test F1 score (%)',
        )
        axes[0][i].plot(
            test_roc_auc_tracker_list[counter],
            color='orange', label='Test ROC AUC (%)',
        )
        axes[0][i].plot(
            test_pr_auc_tracker_list[counter],
            color='pink', label='Test PR AUC (%)',
        )
        axes[0][i].legend()
        axes[0][i].set_xlabel('Epoch iteration')
        axes[0][i].set_ylabel('Accuracy / Value')
        axes[0][i].set_title('Training / Validation metrics')

        cnf_matrix = cnf_matrix_list[counter]
        sns.heatmap(cnf_matrix, annot=True, fmt='d', ax=axes[1][i])
        axes[1][i].set_xlabel('Predicted')
        axes[1][i].set_ylabel('Actual')
        axes[1][i].set_title('Confusion Matrix')

        distributions_train = diagn_dist_list[counter]['Train']
        df_train = pd.DataFrame(
            list(distributions_train.items()), columns=[
                'diagnosis', 'count',
            ],
        )
        sns.barplot(
            data=df_train, x='diagnosis', y='count',
            ax=axes[2][i], palette=color_dict, order=color_dict.keys(),
        )
        axes[2][i].legend(handles=legend_patches)
        axes[2][i].set_title(f'{"Counts of each diagnosis(Train)"}')
        axes[2][i].set_ylabel('Count')
        # Use all diagnoses as x-axis labels
        axes[2][i].set_xticklabels(
            list(color_dict.keys()), rotation=45, ha='right',
        )
        axes[2][i].grid(axis='y')

        distributions_test = diagnosis_pred_distr_list[counter]['Test']
        correct_distributions = diagnosis_pred_distr_list[counter]['Correct']

        # Make sure both dictionaries contain the same set of keys
        for diagnosis in color_dict.keys():
            distributions_test.setdefault(diagnosis, 0)
            correct_distributions.setdefault(diagnosis, 0)

        # Sort both dictionaries by the order of keys in color_dict
        distributions_test = {
            k: distributions_test[k] for k in color_dict.keys()
        }
        correct_distributions = {
            k: correct_distributions[k] for k in color_dict.keys()
        }

        df_test = pd.DataFrame(
            list(distributions_test.items()), columns=[
                'diagnosis', 'count',
            ],
        )
        df_correct = pd.DataFrame(
            list(correct_distributions.items()), columns=[
                'diagnosis', 'correct_count',
            ],
        )

        df_merged = df_test.merge(df_correct, on='diagnosis')

        barplot1 = sns.barplot(
            data=df_merged, x='diagnosis', y='count',
            ax=axes[3][i], palette=color_dict, order=color_dict.keys(),
        )
        barplot2 = sns.barplot(
            data=df_merged, x='diagnosis', y='correct_count',
            ax=axes[3][i], palette=color_dict, hatch='//', order=color_dict.keys(),
        )

        for diagnosis, p in zip(df_merged['diagnosis'], barplot2.patches):
            p.set_facecolor(color_dict[diagnosis])

        axes[3][i].set_title(f'{"Counts of each diagnosis (Test)"}')
        axes[3][i].set_ylabel('Count')
        # Use all diagnoses as x-axis labels
        axes[3][i].set_xticklabels(
            list(color_dict.keys()), rotation=45, ha='right',
        )
        axes[3][i].grid(axis='y')

        counter += 1

    plt.tight_layout()
    plt.show()

    return fig


def plot_metrics_together(train_accuracy_list_tracker, test_accuracy_list_tracker, test_f1_tracker_list):

    # SHould this be std or max/mean?
    fig, ax = plt.subplots(figsize=(5, 5))

    # Calculating the mean and standard deviation for each metric
    mean_train_accuracy = np.mean(train_accuracy_list_tracker, axis=0)
    std_train_accuracy = np.std(train_accuracy_list_tracker, axis=0)

    mean_test_accuracy = np.mean(test_accuracy_list_tracker, axis=0)
    std_test_accuracy = np.std(test_accuracy_list_tracker, axis=0)

    mean_test_f1 = np.mean(test_f1_tracker_list, axis=0)
    std_test_f1 = np.std(test_f1_tracker_list, axis=0)

    # Plotting the mean values
    ax.plot(
        mean_train_accuracy, color='red',
        label='Mean Training accuracy (%)',
    )
    ax.plot(mean_test_accuracy, color='blue', label='Mean Test accuracy (%)')
    ax.plot(mean_test_f1, color='green', label='Mean Test F1 score (%)')

    # Plotting the envelope using the standard deviation
    ax.fill_between(
        range(len(mean_train_accuracy)), mean_train_accuracy -
        std_train_accuracy, mean_train_accuracy+std_train_accuracy, color='red', alpha=0.2,
    )
    ax.fill_between(
        range(len(mean_test_accuracy)), mean_test_accuracy-std_test_accuracy,
        mean_test_accuracy+std_test_accuracy, color='blue', alpha=0.2,
    )
    ax.fill_between(
        range(len(mean_test_f1)), mean_test_f1-std_test_f1,
        mean_test_f1+std_test_f1, color='green', alpha=0.2,
    )

    # Adding legend, x-label, and y-label
    ax.legend()
    ax.set_xlabel('Epoch iteration')
    ax.set_ylabel('Accuracy / Value')
    ax.set_title('Mean Training / Validation metrics with Envelopes')

    # Displaying the plot
    plt.tight_layout()
    plt.show()

    return fig


def filter_and_plot(directory, csv_file):
    """
    Plots multiple metrics together with mean and standard deviation envelopes.

    Args:
        train_accuracy_list_tracker (list of lists): List of training accuracy values for each epoch.
        test_accuracy_list_tracker (list of lists): List of test accuracy values for each epoch.
        test_f1_tracker_list (list of lists): List of test F1 score values for each epoch.

    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure.
    """

    # Load the DataFrame
    df_samples = pd.read_csv(csv_file)

    # Define the directories to check
    dirs_to_check = [
        os.path.join(
            directory, 'sma',
        ), os.path.join(directory, 'non-sma'),
    ]

    # Get the names of all subdirectories in 'sma' and 'non-sma'
    subdir_names = [
        d for dir_to_check in dirs_to_check for d in os.listdir(
            dir_to_check,
        ) if os.path.isdir(os.path.join(dir_to_check, d))
    ]

    # Convert the 'FASt-Mal-Code' column to a set
    fast_mal_code_set = set(df_samples['FASt-Mal-Code'].tolist())

    # Check which subdirectories are not in 'FASt-Mal-Code'
    not_in_df = [
        subdir_name for subdir_name in subdir_names if subdir_name not in fast_mal_code_set
    ]

    for subdir_name in not_in_df:
        modified_name = subdir_name.replace('-', '-0')
        if modified_name in fast_mal_code_set:
            print(f"'{subdir_name}' is not in csv file, but '{modified_name}' is.")
        else:
            print(
                f"Neither '{subdir_name}' nor '{modified_name}' are in the csv file",
            )

    # Check for duplicate names in subdir_names and print them
    subdir_counts = Counter(subdir_names)
    for subdir_name, count in subdir_counts.items():
        if count > 1:
            print(f"'{subdir_name}' is duplicated {count} times in subdir_names")

    # If there are any missing, print them
    if not_in_df:
        print('The following subdirectory names are not in the csv file:')
        for subdir_name in not_in_df:
            print(subdir_name)
    else:
        print('All subdirectory names are in the csv file')

    # Filter df_samples by the names in subdir_names
    df_samples = df_samples[df_samples['FASt-Mal-Code'].isin(subdir_names)]

    # Create a new 'Category' column
    df_samples['Category'] = df_samples['Diagnosis'].apply(
        lambda x: 'SMA' if x.lower() == 'severe malaria anaemia' else 'Non-SMA',
    )

    # Pivot table for category and diagnosis
    df_pivot = df_samples.pivot_table(
        index='Category', columns='Diagnosis', aggfunc='size', fill_value=0,
    )

    # Define color mapping
    color_dict = {
        'Severe Malaria Anaemia': 'red',
        'Malaria, Severe Anaemia, No SMA': 'blue',
        'No Malaria, Anaemia': 'green',
        'No Malaria, Severe Anaemia': 'orange',
        'Malaria, No Anaemia': 'purple',
        'Malaria, Anaemia, No Severe': 'brown',
        'No Malaria, No Anaemia': 'pink',
        'Unclassified': 'grey',
    }

    # Create a color list in the correct order
    color_list = [color_dict[diagnosis] for diagnosis in df_pivot.columns]

    # Plot
    ax = df_pivot.plot(
        kind='bar', stacked=True,
        color=color_list, figsize=(4, 8), edgecolor='black',
    )

    # Add text for counts
    for i, diagnosis in enumerate(df_pivot.columns):
        counts = df_pivot[diagnosis].values
        if i == 0:
            bottoms = [0, 0]
        else:
            bottoms = df_pivot.iloc[:, :i].sum(axis=1).values
        for j, count in enumerate(counts):
            if count > 0:
                plt.text(
                    j, bottoms[j] + count/2, str(count),
                    color='white', ha='center', va='center', weight='bold',
                )

    # Add text on top of each bar for total counts and percentage
    for i in range(df_pivot.shape[0]):
        total_count = df_pivot.iloc[i].sum()
        percentage = total_count / df_samples.shape[0] * 100
        plt.text(
            i, total_count, f'{total_count} samples ({percentage:.0f}%)',
            ha='center', va='bottom', weight='bold',
        )

    plt.ylim(0, 100)  # Set y-axis range from 0 to 100
    plt.title('Count of each class')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Ensure the entire plot fits into the figure area

    legend_labels = ax.legend_.get_texts()
    for i, label in enumerate(legend_labels):
        diagnosis = df_pivot.columns[i]
        total = df_pivot[diagnosis].sum()
        percentage = total / df_samples.shape[0] * 100
        new_label = f'{label.get_text()} ({total} samples, {percentage:.0f}%)'
        label.set_text(new_label)

    # Set legend title to 'Classes' and position it at the top right
    plt.legend(title='Classes', loc='lower center')
    ax.legend(title='Classes', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.show()
