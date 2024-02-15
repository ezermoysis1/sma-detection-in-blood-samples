from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from src.analysis.rbc_comparison_utils import calculate_and_plot_roc_pr_auc
from src.analysis.rbc_comparison_utils import compare_dataframes
from src.analysis.rbc_comparison_utils import display_rbc_comparison
from src.analysis.rbc_comparison_utils import rbc_descriptors_comp
from src.analysis.rbc_comparison_utils import rbf_classification
from src.analysis.rbc_comparison_utils import rbf_n_comparison1

img_path = 'data/rbc_images_cleaned'
model_path = 'Experiment_log/20231020_085650/model_weights_4.pth'

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

# new_df.to_csv('classified_cells_oca.csv', index=False)

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
