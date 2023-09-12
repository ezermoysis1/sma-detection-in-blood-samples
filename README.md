
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

### Data

The dataset used in this paper is The Oxford Pet III dataset. It contains a collection of 7,365 images of cats and dogs and their labels. There are image-level labels, specifying what animal (i.e. dog or cat) and breed is shown, and pixel-level
labels, providing a semantic segmentation, referred to as "ground truth mask", of the input images. Click [here](hhttps://www.robots.ox.ac.uk/~vgg/data/pets/) to download the data.

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