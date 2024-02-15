from __future__ import annotations

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
from PIL import Image
from skimage import color
from skimage import filters
from skimage import io
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from src.models.my_models import my_ResNet_CNN
# from dataloader import CustomImageDataset


class RedCellMorphologyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # resize to 64x64
            transforms.ToTensor(),
        ])

        self.transform_simple = transforms.Compose([
            transforms.Resize((128, 128)),  # resize to 64x64
            transforms.ToTensor(),
        ])

        for label, class_name in enumerate(['non-sma', 'sma']):
            class_dir = os.path.join(root_dir, class_name)
            for subfolder_name in os.listdir(class_dir):
                subfolder_dir = os.path.join(class_dir, subfolder_name)
                for filename in os.listdir(subfolder_dir):
                    # Or whatever format your images are in
                    if filename.endswith('.png'):
                        self.image_paths.append(
                            os.path.join(subfolder_dir, filename),
                        )
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform_simple(img)  # Apply the transform
        # Add the extra batch dimension here
        return img.unsqueeze(0), label, img_path


def rbf_classification(img_path, model_path):

    dataset = RedCellMorphologyDataset(img_path)
    # shuffle=False to keep track of the original order
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = my_ResNet_CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data = []

    with torch.no_grad():
        for imgs, labels, img_paths in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Obtain both predictions and vector features
            outputs, vector_features = model(
                imgs, mode='test', get_features=True,
            )
            probs = outputs.cpu().numpy()
            vector_features = vector_features.cpu().numpy()

            for img_path, label, prob, feature in zip(img_paths, labels, probs, vector_features):
                data.append([img_path, label.item(), prob[0], feature])

    img_class_df = pd.DataFrame(
        data, columns=[
            'image_path', 'true_label', 'predicted_probability', 'vector_features',
        ],
    )

    return img_class_df


def rbf_n_comparison1(img_class_df, num_rbcs_per_class=20):

    # Sort the DataFrame
    img_class_df = img_class_df.sort_values(by='predicted_probability')

    img_class_df = img_class_df[
        img_class_df['true_label']
        == img_class_df['predicted_probability'].round()
    ]
    # Get image paths
    most_non_sma_rbcs = img_class_df['image_path'].head(
        num_rbcs_per_class,
    ).values
    most_sma_rbcs = img_class_df['image_path'].tail(num_rbcs_per_class).values

    return most_non_sma_rbcs, most_sma_rbcs


def rbf_n_comparison(img_class_df, num_rbcs_per_class=20):

    # Sort the DataFrame
    img_class_df = img_class_df.sort_values(by='predicted_probability')

    # img_class_df = img_class_df[img_class_df['true_label'] == img_class_df['predicted_probability'].round()]

    # Filter images within the specified predicted probability ranges
    most_non_sma_rbcs = img_class_df[
        (img_class_df['predicted_probability'] >= 0.01) &
        (img_class_df['predicted_probability'] <= 0.49)
    ]

    most_sma_rbcs = img_class_df[
        (img_class_df['predicted_probability'] >= 0.51) &
        (img_class_df['predicted_probability'] <= 0.99)
    ]

    # Randomly sample num_rbcs_per_class images from each class
    most_non_sma_rbcs = most_non_sma_rbcs.sample(
        n=num_rbcs_per_class, random_state=42,
    )['image_path'].values
    most_sma_rbcs = most_sma_rbcs.sample(n=num_rbcs_per_class, random_state=42)[
        'image_path'
    ].values

    return most_non_sma_rbcs, most_sma_rbcs


def display_rbc_comparison(most_non_sma_rbcs, most_sma_rbcs):

    # Define the number of rows and columns for the grid
    nrows = int(np.round(len(most_non_sma_rbcs) / 4))
    ncols = 4

    # change ncols to ncols*2 and adjust figsize
    fig, ax = plt.subplots(nrows, ncols*2, figsize=(10, nrows))

    # Add title
    fig.suptitle('Red Blood Cells Comparison', fontsize=20)

    # plot 'Non-sma Red Blood Cells'
    for i, image_path in enumerate(most_non_sma_rbcs):
        img = Image.open(image_path)
        img_array = np.array(img)
        ax[i//ncols, i % ncols].imshow(img_array)
        ax[i//ncols, i % ncols].axis('off')  # to remove the axis

    # If the total number of 'Non-sma' images is less than nrows*ncols
    for i in range(len(most_non_sma_rbcs), nrows*ncols):
        ax[i//ncols, i % ncols].axis('off')  # to remove the empty plots

    # plot 'sma Red Blood Cells'
    for i, image_path in enumerate(most_sma_rbcs):
        img = Image.open(image_path)
        img_array = np.array(img)
        # add ncols to the column index for right side plotting
        ax[i//ncols, i % ncols + ncols].imshow(img_array)
        ax[i//ncols, i % ncols + ncols].axis('off')  # to remove the axis

    # If the total number of 'sma' images is less than nrows*ncols
    for i in range(len(most_sma_rbcs), nrows*ncols):
        # to remove the empty plots
        ax[i//ncols, i % ncols + ncols].axis('off')

    # Add a black line between columns 4 and 5, shift the line a little bit to the right, and make it shorter
    line_x = (
        ax[0, ncols-1].get_position().bounds[0] +
        ax[0, ncols].get_position().bounds[0]
    ) / 2
    offset = 0.03  # change this value to move the line more or less
    lower_bound = 0.03  # adjust this value to control the lower bound of the line
    upper_bound = 0.90  # adjust this value to control the upper bound of the line
    plt.plot(
        [line_x + offset, line_x + offset], [lower_bound, upper_bound],
        color='black', transform=plt.gcf().transFigure, clip_on=False, lw=1,
    )

    # add more space between the 4th and 5th columns
    fig.subplots_adjust(wspace=0.3)

    # Add subtitles for the two groups of columns
    x_position_1 = (
        ax[0, 0].get_position().bounds[0] + ax[0, ncols-1].get_position(
        ).bounds[0] + ax[0, ncols-1].get_position().bounds[2]
    ) / 2 - 0.06
    x_position_2 = (
        ax[0, ncols].get_position().bounds[0] + ax[
            0, ncols*2 -
            1,
        ].get_position().bounds[0] + ax[0, ncols*2-1].get_position().bounds[2]
    ) / 2 + 0.035
    y_position = ax[0, 0].get_position().bounds[1] + \
        ax[0, 0].get_position().bounds[3] + 0.03
    # fig.text(x_position_1, y_position, 'non-sma', ha='center', va='bottom', fontsize=14)
    # fig.text(x_position_2, y_position, 'sma', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()  # added to avoid overlapping of titles and images
    plt.show()

    return fig


def get_descriptors_dataframe(image_paths):
    # Initialize an empty list to store each image's descriptor dictionary
    all_descriptors = []

    fig, axs = plt.subplots(5, 4, figsize=(15, 15))
    axs = axs.ravel()

    for idx, image_path in enumerate(image_paths):
        # Check if the file exists
        if not os.path.isfile(image_path):
            print(f'File {image_path} not found.')
            continue

        # Read the image
        image = io.imread(image_path)

        # If the image is not grayscale, convert it to grayscale
        if len(image.shape) > 2:
            gray_image = color.rgb2gray(image)

        # Threshold the image to get a binary image
        threshold_value = filters.threshold_otsu(gray_image)
        binary_image = gray_image < threshold_value

        # Label the image
        label_image = measure.label(binary_image)

        # Use regionprops to get the descriptors
        regions = measure.regionprops(label_image, intensity_image=gray_image)
        # Select the region with the largest area
        region = max(regions, key=lambda region: region.area)

        # For simplicity, we take the descriptors of the first region.
        if regions:
            props = region

            # Shape descriptors calculations
            perimeter = props.perimeter
            area = props.area
            convex_image = props.convex_image
            convex_perimeter = measure.perimeter(convex_image)

            roundness = (4 * np.pi * area) / \
                (convex_perimeter ** 2) if perimeter != 0 else 0
            compactness = perimeter ** 2 / \
                (4 * np.pi * area) if area != 0 else 0
            convexity = convex_perimeter / perimeter if perimeter != 0 else 0

            descriptor_dict = {
                'area': props.area,
                'area_filled': props.filled_area,  # fixed the attribute name
                'equivalent_diameter_area': props.equivalent_diameter,
                'eccentricity': props.eccentricity,
                'convex_area': props.convex_area,
                'extent': props.extent,
                'solidity': props.solidity,
                'perimeter': props.perimeter,
                'perimeter_crofton': props.perimeter_crofton,
                'roundness': roundness,
                'compactness': compactness,
                'convexity': convexity,
            }

            # Draw a rectangle around the chosen region
            minr, minc, maxr, maxc = props.bbox
            rect = patches.Rectangle(
                (minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2,
            )
            axs[idx].imshow(image)
            axs[idx].add_patch(rect)

            # Extract the relevant part of the image path
            title = image_path
            title = title[
                title.index(
                    'non-sma',
                ):
            ] if 'non-sma' in title else title[title.index('sma'):]

            axs[idx].set_title(f'Region from {title}')
            axs[idx].axis('off')
        else:
            descriptor_dict = {
                'area': None,
                'area_filled': None,
                'equivalent_diameter_area': None,
                'eccentricity': None,
                'convex_area': None,
                'extent': None,
                'solidity': None,
                'perimeter': None,
                'perimeter_crofton': None,
                'roundness': None,
                'compactness': None,
                'convexity': None,
            }

        # Add the image path to the dictionary
        descriptor_dict['image_path'] = image_path

        # Add this dictionary to the list
        all_descriptors.append(descriptor_dict)

        # After visualizing 20 images, break the loop
        if idx >= 19:
            break

    # plt.tight_layout()
    # plt.show()

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(all_descriptors)

    return df


def rbc_descriptors_comp(most_non_sma_rbcs, most_sma_rbcs):

    descriptors_highest = get_descriptors_dataframe(most_sma_rbcs)
    descriptors_lowest = get_descriptors_dataframe(most_non_sma_rbcs)

    # Select columns with numeric data
    descriptors_numeric_highest = descriptors_highest.select_dtypes(
        include=[
            np.number,
        ],
    )
    descriptors_numeric_lowest = descriptors_lowest.select_dtypes(
        include=[
            np.number,
        ],
    )

    return descriptors_numeric_highest, descriptors_numeric_lowest


def compare_dataframes(df1, df2):
    # Add an extra column 'Source' for identification
    df1['Source'] = 'SMA'
    df2['Source'] = 'Non-SMA'

    # Concatenate the dataframes
    df = pd.concat([df1, df2])

    # Reshape the dataframe suitable for sns.boxplot
    df_melt = df.melt(id_vars='Source')

    # Get the unique column names (variables)
    columns = df_melt['variable'].unique()
    n_columns = len(columns)

    # Calculate the number of rows and columns for the subplots
    nrows = 3  # 2 columns of subplots
    ncols = 4
    # ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))

    if n_columns == 1:
        # if only one subplot, axes is not an array, this line takes care of that
        axes = [axes]

    for ax, col in zip(axes.flatten(), columns):
        print(col)
        # Create a subset of the data for the current column
        subset = df_melt[df_melt['variable'] == col]

        # T-test
        group1 = subset[subset['Source'] == 'SMA']['value']
        group2 = subset[subset['Source'] == 'Non-SMA']['value']
        t_stat, p_val = stats.ttest_ind(group1, group2)

        # Create a subplot for each column
        sns.boxplot(
            x='variable', y='value', hue='Source',
            data=subset, ax=ax, palette='PRGn',
        )
        # 2 decimal places in scientific notation
        ax.set_title(f'{col} (p-value: {p_val:.2e})')

    # if there are more axes than columns, delete the extra ones
    for i in range(n_columns, nrows*ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

    return fig


def samples_classification(img_path, model_path):
    # Map class to 0/1
    # Dictionary to assign 1 to 'sma' and 0 to 'non-sma' samples
    class_to_idx = {'sma': 1, 'non-sma': 0}

    transform_simple = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),
    ])

    # Initialize your CustomImageDataset
    dataset = CustomImageDataset(
        root_dir=img_path, class_to_idx=class_to_idx, min_number_images=0, transform=transform_simple,
    )

    # Create a data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model and set it to evaluation mode
    model = my_ResNet_CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model = model.to(device)

    data = []

    with torch.no_grad():
        for images, labels, subfolder_names in dataloader:
            # Move each image to the appropriate device
            images = [img.to(device) for img in images]
            labels = labels.to(device)

            # Pass the images through the model up to the feature extraction layer
            # Assuming get_features=True returns vector_features
            outputs, vector_features = model(
                images, mode='test', get_features=True,
            )
            probs = outputs.cpu().numpy()
            vector_features = vector_features.cpu().numpy()

            # Take the mean of vector_features to get one feature for the entire sample
            mean_vector_feature = vector_features.mean(axis=0)

            for label, prob, subfolder_name in zip(labels, probs, subfolder_names):
                data.append([
                    subfolder_name, label.item(),
                    prob[0], mean_vector_feature,
                ])

    # Create a DataFrame with vector features
    features_df = pd.DataFrame(
        data, columns=[
            'subfolder_name', 'true_label', 'predicted_probability', 'vector_features',
        ],
    )

    return features_df


def calculate_and_plot_roc_pr_auc(df):
    true_labels = df['true_label']
    predicted_probabilities = df['predicted_probability']

    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    roc_auc = roc_auc_score(true_labels, predicted_probabilities)

    # PR curve and AUC
    precision, recall, _ = precision_recall_curve(
        true_labels, predicted_probabilities,
    )
    pr_auc = auc(recall, precision)

    # Using a fixed threshold of 0.5 to calculate accuracy, sensitivity, and specificity
    threshold = 0.5
    predictions = predicted_probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Same as recall
    specificity = tn / (tn + fp)

    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (AUC = {roc_auc:.2f})',
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Plot PR curve
    plt.subplot(1, 2, 2)
    plt.plot(
        recall, precision, color='blue', lw=2,
        label=f'PR curve (AUC = {pr_auc:.2f})',
    )
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

    # Print the metrics
    print(f'ROC AUC: {roc_auc:.2f}')
    print(f'PR AUC: {pr_auc:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Sensitivity (Recall or TPR): {sensitivity:.2f}')
    print(f'Specificity (TNR): {specificity:.2f}')
