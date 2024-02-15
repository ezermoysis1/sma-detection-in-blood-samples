from __future__ import annotations

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from tqdm import tqdm

from src.data.dataloader import CustomImageDataset
from src.models.my_models import my_ResNet_CNN


class Experiment:

    def __init__(
        self,
        seed,
        img_dim,
        new_path,
        min_number_images,
        train_ratio,
        batch_size,
        apply_augmentations,
        oversampling,
        class_weight,
        dropout,
        aggregation,
        lr,
    ):
        """
        Initializes an Experiment object with various parameters.

        Args:
            seed (int): Random seed for reproducibility.
            img_dim (int): Image dimension for resizing.
            new_path (str): Path to the dataset directory.
            min_number_images (int): Minimum number of images per class for inclusion in the dataset.
            train_ratio (float): Ratio of data to use for training.
            batch_size (int): Batch size for data loading.
            apply_augmentations (int): Flag for applying data augmentations.
            oversampling (int): Flag for oversampling methods.
            class_weight (int): Flag for applying class weighting.
            dropout (float): Dropout rate for the neural network.
            aggregation (str): Aggregation method for the neural network.
            lr (float): Learning rate for optimization.
        """

        self.seed = seed
        self.img_dim = img_dim
        self.new_path = new_path
        self.min_number_images = min_number_images
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.apply_augmentations = apply_augmentations
        self.oversampling = oversampling
        self.class_weight = class_weight
        self.dropout = dropout
        self.aggregation = aggregation
        self.lr = lr

        # Freeze seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Define the transforms (if augmentations == 0 --> only resize, else apply random transforms)
        self.transform = self._load_transform()

        # Create the dataset and dataloader
        self.train_loader, self.test_loader, dataset = self._create_dataloader()
        print('The Experiment has been created successfully')

        # Print class distribution of dataset
        self.counts = dataset.get_class_counts()
        print('Data consists of the following classes: ', self.counts)

        # Print class distribution of each dataloader
        self.distributions = self.get_loader_class_distribution()
        print('Distribution is: ', self.distributions)

    def _create_dataloader(self):
        """
        Creates training and test data loaders.

        Returns:
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            dataset (CustomImageDataset): CustomImageDataset object.
        """

        # Map class to 0/1
        # Dictionaty to assign 1 to 'sma' and 0 to 'non-sma' samples
        class_to_idx = {'sma': 1, 'non-sma': 0}

        # Create dataset
        dataset = CustomImageDataset(
            root_dir=self.new_path, class_to_idx=class_to_idx,
            min_number_images=self.min_number_images, transform=self.transform,
        )

        # Extract all labels from the dataset
        labels = [label for _, label, _ in dataset]

        # Calculate sizes of each split
        test_size = 1 - (self.train_ratio)

        # Perform stratified splitting
        train_indices, test_indices, _, _ = train_test_split(
            list(
                range(
                    len(
                        dataset,
                    ),
                ),
            ), labels, test_size=test_size, stratify=labels, random_state=self.seed,
        )

        if self.oversampling == 1:  # Resampling
            # Oversampling
            ros = RandomOverSampler(random_state=self.seed)

            # Apply the transform to the training set
            train_indices_resampled, _ = ros.fit_resample(
                np.array(train_indices).reshape(
                    -1,
                    1,
                ), [labels[i] for i in train_indices],
            )

            train_indices = train_indices_resampled.ravel()

            # Create data loaders for each split
            train_loader = DataLoader(
                Subset(dataset, train_indices), batch_size=self.batch_size, shuffle=True,
            )
            test_loader = DataLoader(
                Subset(dataset, test_indices), batch_size=self.batch_size, shuffle=False,
            )

        elif self.oversampling == 2:  # SMOTE
            # Initialize SMOTE
            smote = SMOTE(random_state=self.seed)

            # Apply SMOTE to the training set
            train_indices_resampled, _ = smote.fit_resample(
                np.array(train_indices).reshape(
                    -1,
                    1,
                ), [labels[i] for i in train_indices],
            )

            train_indices = train_indices_resampled.ravel()

            # Create data loaders for each split
            train_loader = DataLoader(
                Subset(dataset, train_indices), batch_size=self.batch_size, shuffle=True,
            )
            test_loader = DataLoader(
                Subset(dataset, test_indices), batch_size=self.batch_size, shuffle=False,
            )

        if self.oversampling == 3:  # Adasyn
            # Oversampling
            ada = ADASYN(random_state=self.seed)

            # Apply the transform to the training set
            train_indices_resampled, _ = ada.fit_resample(
                np.array(train_indices).reshape(
                    -1,
                    1,
                ), [labels[i] for i in train_indices],
            )

            train_indices = train_indices_resampled.ravel()

            # Create data loaders for each split
            train_loader = DataLoader(
                Subset(dataset, train_indices), batch_size=self.batch_size, shuffle=True,
            )
            test_loader = DataLoader(
                Subset(dataset, test_indices), batch_size=self.batch_size, shuffle=False,
            )

        else:
            # Create data loaders for each split
            train_loader = DataLoader(
                torch.utils.data.Subset(
                    dataset, train_indices,
                ), batch_size=self.batch_size, shuffle=True,
            )
            test_loader = DataLoader(
                torch.utils.data.Subset(
                    dataset, test_indices,
                ), batch_size=self.batch_size, shuffle=False,
            )

        return train_loader, test_loader, dataset

    def train(self, epochs, bag_size, class_weight):
        """
        Trains a neural network on the dataset.

        Args:
            epochs (int): Number of training epochs.
            bag_size (int): Bag size for neural network.
            class_weight (int): Flag for applying class weighting.

        Returns:
            None
        """

        pos_weight = self._calculate_weight(class_weight)

        print('The weight applied to the positive class is: ', pos_weight.item())

        # Initialize the network, loss and optimizer
        net = my_ResNet_CNN(dropout=self.dropout, aggregation=self.aggregation)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(net.parameters(), lr=self.lr)

        # Check if GPU is available and print in what device training will take place
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('This will run on', device)

        # Initialize lists to store accuracy values
        self.train_accuracy_list = []
        self.test_accuracy_list = []
        test_accuracy = 0.0
        loss_tracker = []
        self.test_f1_tracker = []
        self.test_roc_auc_tracker = []
        self.test_pr_auc_tracker = []

        # Training loop
        net.train()

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            correct = 0
            total = 0

            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{1}', unit='batch') as pbar:
                for i, data in enumerate(self.train_loader, 0):

                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels, _ = data

                    # Move each tensor in the list, all labels, and the model to the GPU
                    inputs = [tensor.to(device) for tensor in inputs]
                    # print('inputs size: ', inputs.size())

                    labels = labels.to(device)
                    net = net.to(device)
                    loss_fn.to(device)

                    # Make a prediciton by passing inputs sample through network
                    target = net(inputs, mode='train', bag_size=bag_size)

                    # Calculate the loss
                    loss = loss_fn(target[0], labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_tracker.append(loss)

                    # Update accuracy
                    predicted = target.data.round().int()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # print statistics
                    running_loss += loss.item()

                    # Print loss in every 10 epochs
                    pbar.update(1)

            # Calculate training accuracy
            train_accuracy = 100 * correct / total

            self.train_accuracy_list.append(train_accuracy)

            # Test metrics - store and print test metrics at the end of each epoch
            net.eval()  # Switch to evaluation mode
            test_correct = 0
            test_total = 0
            test_act_labels = []
            test_pred_labels = []
            test_pred_probs = []
            diagnosis_correct_counts = defaultdict(int)
            diagnosis_total_counts = defaultdict(int)
            self.preds_record = []

            with torch.no_grad():
                for data in self.test_loader:
                    inputs, labels, id = data

                    inputs = [tensor.to(device) for tensor in inputs]

                    # Make prediction for test data
                    target = net(inputs, mode='eval')
                    target = target.to(device)

                    # Load the actual labels
                    labels = labels.to(device)

                    # Calculate test accuracy
                    predicted = target.data.round().int()
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

                    # Calculate F1 score for test set
                    test_predicted_probs = target.cpu().numpy()
                    test_predicted_labels = np.round(test_predicted_probs, 0)

                    if epoch == epochs-1:
                        # Get the corresponding diagnosis
                        id = id[0]  # extract string from tuple
                        diagnosis = self.df_samples[
                            self.df_samples['FASt-Mal-Code']
                            == id
                        ]['Diagnosis'].values[0]

                        # Check if the prediction is correct
                        correct = (predicted == labels).sum().item()

                        # Update the diagnosis correct and total counts
                        diagnosis_correct_counts[diagnosis] += correct
                        diagnosis_total_counts[diagnosis] += labels.size(0)

                        # create a dictionary for the current iteration
                        current_preds_record = {
                            'seed': self.seed,
                            'FASt-Mal-Code': id,
                            'Diagnosis': diagnosis,
                            'true label': labels.cpu().numpy().item(),
                            'predicted_label': int(test_predicted_labels.item()),
                            'predicted probability': test_predicted_probs[0][0],
                        }

                        self.preds_record.append(current_preds_record)

                    test_act_labels.append(labels.cpu().numpy().item())
                    test_pred_labels.append(int(test_predicted_labels.item()))
                    test_pred_probs.append(test_predicted_probs[0][0])

            if epoch == epochs-1:

                # Create the final distribution dictionary
                self.diagnosis_distribution = {
                    'Test': diagnosis_total_counts,
                    'Correct': diagnosis_correct_counts,
                }

            # Calculate accuracy for test set
            test_accuracy = 100 * test_correct / test_total

            # Calculate F1 score for test set
            test_f1 = f1_score(test_act_labels, test_pred_labels)

            # Calculate ROC AUC for test set
            fpr, tpr, _ = roc_curve(test_act_labels, test_pred_probs)
            test_roc_auc = auc(fpr, tpr)

            # Calculate PR AUC for test set
            test_pr_auc = average_precision_score(
                test_act_labels, test_pred_probs,
            )

            # Store metrics in lists to plot afterwards
            self.test_accuracy_list.append(test_accuracy)
            self.test_f1_tracker.append(test_f1*100)
            self.test_roc_auc_tracker.append(test_roc_auc*100)
            self.test_pr_auc_tracker.append(test_pr_auc*100)
            # Convert data_records into a pandas DataFrame

            # Print validation metrics
            print('loss                : ',  running_loss)
            print('Training accuracy   : ', train_accuracy)
            print('Test accuracy       : ', test_accuracy)
            print('Test F1 score       : ', test_f1 * 100)
            print('Test ROC AUC        : ', test_roc_auc * 100)
            print('Test PR AUC         : ', test_pr_auc * 100)

        print('Training Completed')

        # Save the trained model weights in a specific folder
        model_path = 'Experiments_log/model_weights_' + str(self.seed) + '.pth'
        torch.save(net.state_dict(), model_path)
        print('Model saved')

        # Compute confusion matrix
        self.cnf_matrix = confusion_matrix(test_act_labels, test_pred_labels)

    def return_metrics(self):
        """
        Returns various metrics and data collected during training.

        Returns:
            Tuple: A tuple containing training accuracy list, test accuracy list, test F1 score tracker,
            test ROC AUC tracker, test PR AUC tracker, confusion matrix, diagnosis distribution, and predictions record.
        """

        return self.train_accuracy_list, self.test_accuracy_list, self.test_f1_tracker, self.test_roc_auc_tracker, self.test_pr_auc_tracker, self.cnf_matrix, self.diagnosis_distribution, self.preds_record

    def plot_metrics(self):
        """
        Plots training and test metrics.

        Returns:
            None
        """

        # Plotting the accuracy curves for this runexperiment.g
        plt.plot(
            self.train_accuracy_list, color='red',
            label='Training accuracy (%)',
        )
        plt.plot(
            self.test_accuracy_list, color='blue',
            label='Test accuracy (%)',
        )
        plt.plot(
            self.test_f1_tracker, color='green',
            label='Test F1 score (%)',
        )
        plt.plot(
            self.test_roc_auc_tracker,
            color='orange', label='Test ROC AUC (%)',
        )
        plt.plot(
            self.test_pr_auc_tracker,
            color='pink', label='Test PR AUC (%)',
        )

        # Adding legend, x-label, and y-label
        plt.legend()
        plt.xlabel('Epoch iteration')
        plt.ylabel('Accuracy / Value')
        plt.title('Training / Test metrics')

        # Displaying the plot
        plt.show()

    def plot_cnf(self):
        """
        Plots the confusion matrix.

        Returns:
            None
        """

        # Print confusion matrix
        print('Confusion Matrix: ')
        print(self.cnf_matrix)

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(self.cnf_matrix, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Displaying the plot
        plt.show()

    def get_class_counts(self):
        """
        Returns the class counts in the dataset.

        Returns:
            dict: A dictionary containing class counts.
        """

        return self.counts

    def get_loader_class_distribution(self):
        """
        Returns the class distribution of training and test data loaders.

        Returns:
            dict: A dictionary containing class distributions.
        """

        loaders = [self.train_loader, self.test_loader]
        loader_names = ['Train', 'Test']
        distributions = {}

        for loader, name in zip(loaders, loader_names):
            total = 0
            class_counts = {0: 0, 1: 0}

            for _, labels, subfolder_name in loader:
                labels = labels.tolist()  # convert tensor to list
                class_counts[0] += labels.count(0)
                class_counts[1] += labels.count(1)
                total += len(labels)

            distributions[name] = {
                'total': total,
                'class_counts': class_counts,
            }

        return distributions

    def get_loader_diagnosis_distribution(self):
        """
        Returns the diagnosis distribution of training and test data loaders.

        Returns:
            dict: A dictionary containing diagnosis distributions.
        """

        self.df_samples = pd.read_csv(
            'samples_dataset.csv',
        )  # load the dataframe

        loaders = [self.train_loader, self.test_loader]
        loader_names = ['Train', 'Test']
        distributions = {}

        for loader, name in zip(loaders, loader_names):
            subfolder_names = []  # to store all subfolder names for this loader

            for _, _, subfolder_name in loader:
                subfolder_name = subfolder_name[0]  # extract string from tuple
                subfolder_names.append(subfolder_name)

            # print(subfolder_names)

            # filter df_samples by the matching 'FASt-Mal-Code' and count the 'Diagnosis'
            diagnosis_counts = self.df_samples[
                self.df_samples['FASt-Mal-Code'].isin(
                    subfolder_names,
                )
            ]['Diagnosis'].value_counts().to_dict()
            distributions[name] = diagnosis_counts

        # self.plot_distribution(distributions)

        return distributions

    def plot_distribution(self, distributions):
        """
        Plots the diagnosis distribution.

        Args:
            distributions (dict): A dictionary containing diagnosis distributions.

        Returns:
            None
        """

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        for i, (name, diagnosis_counts) in enumerate(distributions.items()):
            df = pd.DataFrame(
                list(diagnosis_counts.items()),
                columns=['diagnosis', 'count'],
            )
            sns.barplot(data=df, x='diagnosis', y='count', ax=axs[i])

            axs[i].set_title(f'Counts of each diagnosis ({name})')
            axs[i].set_xlabel('Diagnosis')
            axs[i].set_ylabel('Count')
            # Rotate labels on the x-axis for better visibility
            axs[i].tick_params(axis='x', rotation=45)
            axs[i].grid(axis='y')  # add horizontal grid lines

        plt.tight_layout()
        plt.show()

    def _load_transform(self):
        """
        Loads and returns the data transformation.

        Returns:
            torchvision.transforms.Compose: Composed data transformation.
        """

        transform_augm = transforms.Compose([
            transforms.Resize((self.img_dim, self.img_dim)),
            transforms.RandomRotation(90),
            transforms.ColorJitter(),
            transforms.ToTensor(),
        ])

        transform_simple = transforms.Compose([
            transforms.Resize((self.img_dim, self.img_dim)),  # resize to 64x64
            transforms.ToTensor(),
        ])

        if self.apply_augmentations == 1:
            # Create a composed transform with some % probability for each transform
            composed_transform = transform_augm

        elif self.apply_augmentations == 0:
            composed_transform = transform_simple

        return composed_transform

    def _calculate_weight(self, class_weight):
        """
        Calculates the class weight based on the dataset distribution.

        Args:
            class_weight (int): Flag for applying class weighting.

        Returns:
            torch.Tensor: Positive class weight.
        """

        # Option 1: Class weighting
        pos_weight = torch.tensor([1])  # Default class weight
        distribution = self.get_loader_class_distribution()

        # Dealing with imbalanced dataset
        if class_weight == 1:
            sma_num = distribution['Train']['class_counts'][1]
            non_sma_num = distribution['Train']['class_counts'][0]
            pos_weight = torch.tensor([non_sma_num/sma_num])

        return pos_weight
