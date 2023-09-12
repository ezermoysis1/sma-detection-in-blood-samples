
# MSc Dissertation: Detecting Severe Malaria Anaemia and investigating the morphological characteristics of red blood cells at its presence

## Project Description

UCL MSc Final Project

Date: Sept 2023

In this project, the hypothesis is that in SMA infected patients, RBCs are degraded or ’pitted’ in the spleen due to the spleen’s deterioration, and consequently, there are morphological differences between RBCs of SMA positive and SMA negative patients. The aim of this study is to:
- test the hypothesis through training convolutional neural networks to detect the presence of SMA from a bag/ensemble of thin blood cell sample images and
- describe the differences in the morphological characteristics of the red blood cells in SMA
vs non-SMA patients through a systematic comparison of cells that the model classifies as SMA vs. non-SMA with highest certainty.

![Project Logo](./images/exp1_vfff.PNG)

- Experiment 2: Two additional models were trained on 5%/95% and 10%/90% labeled/unlabelled splits, using Mean Teacher, with the objective to understand how reducing the number of labeled data in Mean Teacher models, affects their performance. Comparative performance of all the models trained is shown below:
![Project Logo](./images/my_pics.png)

## Setup

### Setting up a virtual environment
First, clone the repository:

```bash
git clone https://github.com/ezermoysis1/mean-teacher-ssl
```

Change your directory to where you cloned the files:

```bash
cd mean-teacher-ssl
```

Create a virtual environment with Python 3.6 or above:

```bash
virtualenv venv --python=python3.7 (or python3.7 -m venv venv or conda create -n multiqa python=3.7)
```

Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use the project.

```bash
source venv/bin/activate (or source venv/bin/activate.csh or conda activate multiqa)
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
    
## Use the code

### Dataset

#### Clinical malaria microscopy

Thin Blood Films (TBFs) are first stained with Giemsa at clinics in the University College Hospital (UCH) in the city of Ibadan, Nigeria. Malaria affected cells are detected and counted by human-expert microscopists. A patient is declared malaria positive, if at least one malaria affected erythrocyte (i.e. red blood cell with malaria parasite) is detected in 100 high magnification (100x) TBF Field of Views (FoVs). In addition, a patient is declared Severe Malaria Anaemia (SMA) positive if they are malaria positive and have Packed Cell Volume (PCV) percentage lower than 16$\%$. PCV is clinically a good proxy for measuring level of haemoglobin (Hb) concentration \citep{Turkson2015}. Based on the PCV concentration, SMA negative patients are sub-classified based on presence of malaria and or anaemia as discussed in Section \ref{chap: intro}. The corresponding films are then digitized, processed as discussed in Section \ref{chap: methods}, and used to train and evaluate our MILSMA models. 

#### Data Acquisition

Images from Giemsa-stained thin blood smears are obtained using an Olympus BX63 upright brightfield microscope equipped with a 100X/1.4NA lens, a Prior Scientific motorized stage, and an Edge 5.5c, PCO color camera. The captured image from each field spans 166$\mu$m x 142$\mu$m, translating to a resolution of 2560x2160 pixels. For every position, a z-stack of 14 different focal levels, distanced at 0.5$\mu$m intervals, is recorded with an exposure duration of 50ms. These z-stacks are then merged into one single plane using a wavelet-enhanced depth of field method.

![Project Logo](./Images/sma_whole_slide.png) ![Project Logo](./Images/non_sma_whole_slide.png)

<style>
.image-container {
    display: flex;
    justify-content: space-between; /* Adjust this property for spacing between images */
    align-items: center; /* Vertically center images if needed */
}

.image-container img {
    max-width: 45%; /* Adjust the width of the images */
    height: auto;
}
</style>

<div class="image-container">
    <img src="./Images/sma_whole_slide.png" alt="Image 1">
    <img src="./Images/non_sma_whole_slide.png" alt="Image 2">
</div>


#### Data sets

Original Dataset: The entire dataset used for the scope of this project consists of 128 samples of TBF FoV images. These images will also be referred to in this report as Whole-Slide images (WSI). For each sample, 3 to 20 WSIs have been acquired. Most samples have 5 or 10 images, with some having up to 20. In total, the WSIs in the dataset add up to 1,207. All these images have a size of [2160,2560,3]. The split of non-SMA and SMA samples is 95 (74%) / 33 (26%). Once RBC segmentation is performed, 15,178 RBC images are extracted from the WSIs.

Imbalanced Dataset: After data curation of the RBC segmented images of the original dataset is performed, the resulting dataset is one of the two datasets that are used in training, the imbalanced dataset. This consists of 104 samples with an imbalanced non-SMA and SMA split of 75 (72%) / 29 (28%). This dataset consists of 10,638 RBC images. 

Balanced Dataset: A balanced version of the imbalanced dataset is then created, by randomly selecting 29 non-SMA samples and keeping all the SMA samples from the imbalanced dataset. This will be referred to as the balanced dataset. This dataset consists of 5,837 RBC images and is used to compare how the performance metrics of models trained with each of the two datasets differ. 

The three datasets and the process for obtaining the two last ones are visually described in the Figure below. In the same figure, the breakdown of SMA negative samples (or Non-SMA as it appears in the figure) into sub-classes is also provided. These sub-classes (Malaria & Anaemia No severe, Malaria & No Amaemia, Malaria & Severe Anaemia No SMA, No Malaria & Anaemia, No Malaria & No Anaemia, No Malaria & Severe Anaemia, Unclassified and SMA) are given based on the clinical diagnosis and take into account the presence of parasitemia and PCV count. These sub-classes provide a deeper understanding of the SMA negative class.

Comment on size of dataset: It is worth highlighting that the amount of data that is included in this study is relatively small compared to other studies of automatic malaria detection. In particular, given the complexity of the task and the machine learning techniques used to train the models, a much larger number of samples should be used, and especially for the underrepresented SMA class.

![Project Logo](./Images/dataset.png)


#### Ethical Statement

The internationally recognized ethics committee at the Institute for Advanced Medical Research and Training (IAMRAT) of the College of Medicine, University of Ibadan (COMUI) approved this research with permit numbers: UI/EC/10/0130, UI/EC/19/0110. Parents and/or guardians of study participants gave informed written consent in accordance with the World Medical Association ethical principles for research involving human subjects.

[here](hhttps://www.robots.ox.ac.uk/~vgg/data/pets/) 

### Train

To train a model with with L = 25% labelled and (1-L) = 75% unlabelled data (referred to as 'M-25' in the report) use the following script. This also trains two benchmark supervised models only, with L% labelled data (referred to as 'M25L' or 'M-25L' in the report, and 100% data respectively ('MU' or M-100'). 

```bash
python main_pt.py 0.25
```

To train models with different labelled/unlabelled data splits, change the float after the .py. For example to train a L=35% and the two benchmark models, use the following

```bash
python main_pt.py 0.35
```

### Evaluation 

To evaluate all of the trained models run the following:

```bash
python main_pt.py evaluate
```

## Authors

- [@ezermoysis1](https://github.com/ezermoysis1)
- [@fclarke1](https://github.com/fclarke1)

## Documentation
Please read the full report of the project [here](https://drive.google.com/file/d/1zX3HGt0AiCVF5MfM4lKS9Ag_boOhq-_c/view?usp=sharing)