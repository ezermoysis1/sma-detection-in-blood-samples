from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.offsetbox import OffsetImage
from PIL import Image
from sklearn.decomposition import PCA

from src.analysis.rbc_comparison_utils import calculate_and_plot_roc_pr_auc
from src.analysis.rbc_comparison_utils import compare_dataframes
from src.analysis.rbc_comparison_utils import display_rbc_comparison
from src.analysis.rbc_comparison_utils import rbc_descriptors_comp
from src.analysis.rbc_comparison_utils import rbf_classification
from src.analysis.rbc_comparison_utils import rbf_n_comparison1

img_path = 'data/rbc_images_cleaned'
model_path = 'logs/20231020_085650/model_weights_4.pth'

img_class_df = rbf_classification(img_path, model_path)

# Calculate and plot ROC, PR, and AUC
calculate_and_plot_roc_pr_auc(img_class_df)

# Plot PCA ALL

# Extract the 1000-dimensional vector features, true labels, and predicted probabilities
X = img_class_df['vector_features'].values.tolist()
y_true = img_class_df['true_label'].values.tolist()
y_pred_prob = img_class_df['predicted_probability'].values.tolist()

# Round the predicted probabilities to obtain predicted labels
y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit and transform the features to 2 dimensions
X_2d = pca.fit_transform(X)

# Create a new DataFrame with the reduced features, true labels, and predicted labels
pca_df = pd.DataFrame(data=X_2d, columns=['PCA1', 'PCA2'])
pca_df['true_label'] = y_true
pca_df['predicted_label'] = y_pred

# Separate data by TP, TN, FP, and FN
TP = pca_df[(pca_df['true_label'] == 1) & (pca_df['predicted_label'] == 1)]
TN = pca_df[(pca_df['true_label'] == 0) & (pca_df['predicted_label'] == 0)]
FP = pca_df[(pca_df['true_label'] == 0) & (pca_df['predicted_label'] == 1)]
FN = pca_df[(pca_df['true_label'] == 1) & (pca_df['predicted_label'] == 0)]

# Create a scatter plot with different colors for TP, TN, FP, and FN
plt.figure(figsize=(10, 6))
plt.scatter(TP['PCA1'], TP['PCA2'], label='TP', c='green', alpha=0.5)
plt.scatter(TN['PCA1'], TN['PCA2'], label='TN', c='blue', alpha=0.5)
plt.scatter(FP['PCA1'], FP['PCA2'], label='FP', c='red', alpha=0.5)
plt.scatter(FN['PCA1'], FN['PCA2'], label='FN', c='purple', alpha=0.5)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.title('PCA Visualization with TP, TN, FP, and FN')
plt.show()

# Plot PCA 2 classes

# Extract the 1000-dimensional vector features and true labels
X = img_class_df['vector_features'].values.tolist()
y = img_class_df['true_label'].values.tolist()

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit and transform the features to 2 dimensions
X_2d = pca.fit_transform(X)

# Create a new DataFrame with the reduced features
pca_df = pd.DataFrame(data=X_2d, columns=['PCA1', 'PCA2'])
pca_df['true_label'] = y

# Separate data by class
class_0 = pca_df[pca_df['true_label'] == 0]
class_1 = pca_df[pca_df['true_label'] == 1]

# Create a scatter plot to visualize the distinctions
plt.figure(figsize=(10, 6))
plt.scatter(class_0['PCA1'], class_0['PCA2'], label='Class 0', alpha=0.5)
plt.scatter(class_1['PCA1'], class_1['PCA2'], label='Class 1', alpha=0.5)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.title('PCA Visualization of Vector Features')
plt.show()

explained_variance = pca.explained_variance_ratio_

total_variance_captured_by_2_pcs = sum(explained_variance[:2])

print(
    f'Total variance captured by the first 2 PCs: {total_variance_captured_by_2_pcs * 100:.2f}%',
)


# Drop the 'vector_features' column from img_class_df
new_df = img_class_df.drop(columns=['vector_features'])

# Add the two PCA features to the new DataFrame
new_df['PCA1'] = pca_df['PCA1']
new_df['PCA2'] = pca_df['PCA2']

df = new_df

most_non_sma_rbcs, most_sma_rbcs = rbf_n_comparison1(
    img_class_df, num_rbcs_per_class=40,
)
fig_rbc_img_comp = display_rbc_comparison(most_non_sma_rbcs, most_sma_rbcs)

descriptors_numeric_highest, descriptors_numeric_lowest = rbc_descriptors_comp(
    most_non_sma_rbcs, most_sma_rbcs,
)

fig_rbc_desc_comp = compare_dataframes(
    descriptors_numeric_highest, descriptors_numeric_lowest,
)

# Sort the dataframe by predicted probability in ascending order
df_sorted_asc = new_df.sort_values(
    by=['predicted_probability'], ascending=True,
)

# Get the 50 images with lowest probability
low_prob_images = df_sorted_asc[:50]['image_path'].tolist()
low_prob_probs = df_sorted_asc[:50]['predicted_probability'].tolist()
low_prob_labels = df_sorted_asc[:50]['true_label'].tolist()

# Sort the dataframe by predicted probability in descending order
df_sorted_desc = df.sort_values(by=['predicted_probability'], ascending=False)

# Get the 50 images with highest probability
high_prob_images = df_sorted_desc[:50]['image_path'].tolist()
high_prob_probs = df_sorted_desc[:50]['predicted_probability'].tolist()
high_prob_labels = df_sorted_desc[:50]['true_label'].tolist()


# Function to plot images on the PCA scatter plot
def imscatter(x, y, image_paths, ax=None, zoom=1, colors=None):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, path, color in zip(x, y, image_paths, colors):
        try:
            img = Image.open(path).convert('RGBA')
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(
                image, (x0, y0), xycoords='data', frameon=True,
                bboxprops=dict(edgecolor=color),
            )
            artists.append(ax.add_artist(ab))
        except OSError:
            print(f'Error opening image at {path}.')
    return artists


# Color mapping for true_label
color_map = {0: 'green', 1: 'red'}  # Adjust colors as needed

# Apply color mapping
df['color'] = df['true_label'].map(color_map)

# Sort by predicted_probability in ascending order and select the lowest 50
lowest_50 = df.nsmallest(30, 'predicted_probability')

# Sort by predicted_probability in descending order and select the highest 50
highest_50 = df.nlargest(30, 'predicted_probability')

# Plotting
fig, ax = plt.subplots(figsize=(15, 9))

# Plot all the points as scatter plot with colors based on true_label
ax.scatter(df['PCA1'], df['PCA2'], c=df['color'], alpha=0.5)

# Highlight the 50 lowest probabilities with images
imscatter(
    lowest_50['PCA1'], lowest_50['PCA2'],
    lowest_50['image_path'], ax=ax, zoom=0.2, colors=lowest_50['color'],
)

# Highlight the 50 highest probabilities with images
imscatter(
    highest_50['PCA1'], highest_50['PCA2'],
    highest_50['image_path'], ax=ax, zoom=0.2, colors=highest_50['color'],
)

# Additional plot formatting
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_title(
    'Cell Images with Lowest and Highest Predicted Probabilities Colored by True Label',
)
plt.show()

# Extract 'sample_name' from 'image_path'
df['sample_name'] = df['image_path'].str.extract(r'sma/(.*?)/')

# Assuming 'predicted_probability' is the name of the column in your DataFrame
threshold = 0.5
df['predicted_label'] = df['predicted_probability'].apply(
    lambda x: 1 if x > threshold else 0,
)

# Create a new DataFrame result_df
result_df = df.groupby('sample_name').agg({
    'true_label': 'mean',
}).reset_index()

# Rename the 'true_label' column to 'average_true_label'
result_df = result_df.rename(columns={'true_label': 'true_label'})

# Merge with the original DataFrame to ensure correct alignment
result_df = result_df.merge(
    df[df['predicted_probability'].round() == 1].groupby(
        'sample_name',
    ).size().reset_index(name='count_sma_cells'),
    on='sample_name', how='left',
)


result_df = result_df.merge(
    df[df['predicted_probability'].round() == 0].groupby(
        'sample_name',
    ).size().reset_index(name='count_nonsma_cells'),
    on='sample_name', how='left',
)

# Calculate the percentage of sma cells
result_df['percentage_of_sma_cells'] = (
    result_df['count_sma_cells'] / (
        result_df['count_sma_cells'] + result_df['count_nonsma_cells']
    )
) * 100

# Fill NaN values with 0 (in case there are no instances of sma or nonsma cells)
result_df = result_df.fillna(0)


# Assuming your DataFrame is named df
# Group by 'avg_actual_label' and calculate mean and standard deviation of 'percentage_of_sma_cells'
grouped_df = result_df.groupby('true_label')['percentage_of_sma_cells'].agg([
    'mean', 'std',
]).reset_index()

# Load the 'paper_data_analysis.csv' into a dataframe called df_a
df_a = pd.read_csv('paper_data_analysis.csv')

# Assuming you already have a dataframe named result_df
# Merge result_df with df_a based on the 'sample_name' column
result_df_a = result_df.merge(
    df_a[['FASt-Mal-Code', 'PCV']],
    left_on='sample_name', right_on='FASt-Mal-Code', how='left',
)

# Rename the 'PCV' column in result_df to avoid conflicts
result_df_a.rename(
    columns={
        'PCV_x': 'PCV_result_df',
        'PCV_y': 'PCV_df_a',
    }, inplace=True,
)

# Merge result_df with df_a based on the 'sample_name' column
result_df_a = result_df_a.merge(
    df_a[['FASt-Mal-Code', 'SMA']], left_on='sample_name', right_on='FASt-Mal-Code', how='left',
)

# Merge result_df with df_a based on the 'sample_name' column
result_df_a = result_df_a.merge(
    df_a[['FASt-Mal-Code', 'Diagnosis']], left_on='sample_name', right_on='FASt-Mal-Code', how='left',
)

result_df_a = result_df_a.drop('FASt-Mal-Code_x', axis=1)
result_df_a = result_df_a.drop('FASt-Mal-Code_y', axis=1)
result_df_a['MATCH'] = np.where(
    ((result_df_a['SMA'] == 'NO') & (result_df_a['true_label'] == 0)) | (
        (result_df_a['SMA'] == 'SMA') & (result_df_a['true_label'] == 1)
    ), True, False,
)
result_df_a
result_df_a['SMA'] = result_df_a['SMA'].replace({'SMA': 'SMA+', 'NO': 'SMA-'})
result_df_a = result_df_a.drop('FASt-Mal-Code', axis=1)

result_df_a.to_csv('paper_samples_final.csv', index=False)


result_df_a = result_df_a[result_df_a['Diagnosis'] != 'Unclassified']

result_df_a['Diagnosis'] = result_df_a['Diagnosis'].replace(
    'No Malaria, No Anaemia', 'Malaria Negative, No Anaemia',
)

result_df_a['Diagnosis'] = result_df_a['Diagnosis'].replace(
    'No Malaria, Anaemia', 'Malaria Negative, Anaemia',
)

result_df_a['Diagnosis'] = result_df_a['Diagnosis'].replace(
    'No Malaria, Severe Anaemia', 'Malaria Negative, Severe Anaemia',
)

result_df_a['Diagnosis'] = result_df_a['Diagnosis'].replace(
    'Malaria, No Anaemia', 'Malaria Positive, No Anaemia',
)

result_df_a['Diagnosis'] = result_df_a['Diagnosis'].replace(
    'Severe Malaria Anaemia', 'Malaria Positive, Severe Malaria Anaemia',
)

distinct_diagnosis_values = result_df_a['Diagnosis'].unique()

for value in distinct_diagnosis_values:
    print(value)

result_df_a = result_df_a.dropna()


# Create a dictionary to map the unique diagnosis values to colors
diagnosis_colors = {
    'Malaria Negative, Severe Anaemia': 'red',
    'Malaria Negative, No Anaemia': 'orange',
    'Malaria Negative, Anaemia': 'pink',
    'Malaria Positive, No Anaemia': 'green',
    'Malaria Positive, Anaemia (no-SMA)': 'purple',
    'Malaria Positive, Severe Malaria Anaemia': 'blue',
}

# Define the correct order of classes
diagnosis_order = [
    'Malaria Negative, No Anaemia',
    'Malaria Negative, Anaemia',
    'Malaria Negative, Severe Anaemia',
    'Malaria Positive, No Anaemia',
    'Malaria Positive, Anaemia (no-SMA)',
    'Malaria Positive, Severe Malaria Anaemia',
]

# Filter the DataFrame based on 'Diagnosis' column values
filtered_df = result_df_a[
    result_df_a['Diagnosis'].isin(
        diagnosis_colors.keys(),
    )
]

# Set the style for better aesthetics (optional)
sns.set(style='whitegrid')

# Box Plot with specified order
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Diagnosis', y='percentage_of_sma_cells',
    data=filtered_df, palette=diagnosis_colors, order=diagnosis_order,
)
plt.title('Percentage of SMA cells per sample for different Clinical cases')

# Adjust x-axis labels with new lines
plt.xticks(rotation=0, fontsize=8)  # Set rotation to 45 and font size to 10
ax = plt.gca()
labels = ax.get_xticklabels()
# Split the text at comma and start a new line
new_labels = [label.get_text().replace(', ', ',\n') for label in labels]
ax.set_xticklabels(new_labels)

# Set y-axis title
plt.ylabel('% of SMA cells within each sample')

# Set x-axis title
plt.xlabel('Clinical cases')

plt.show()
