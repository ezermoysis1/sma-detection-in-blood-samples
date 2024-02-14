from __future__ import annotations

import os
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage import filters
from skimage import io
from skimage import measure


def detect_and_calculate_area_ratio(image):
    """
    Given a PIL.Image object, this function detects white object(s) against a black
    background in a binary image, calculates their area, and returns the area ratio
    of the white object(s) to the total image area.
    """

    # Convert the PIL Image to an OpenCV usable format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image (assuming objects are white and the background is black)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    # Calculate the total area of the contours (objects)
    total_object_area = sum(cv2.contourArea(cnt) for cnt in contours)

    # Calculate the total image area
    total_image_area = image.shape[0] * image.shape[1]

    # Calculate and return the area ratio
    return total_object_area / total_image_area


def gather_images(directory):
    # Stores the paths of .png images
    png_paths = []

    # Stores the paths of .png images where the area of the object depicted
    # is more than 0.7 of the total image or less than 0.1
    target_images = []

    # Walk through all directories and files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                png_paths.append(path)

                # Open the image file
                with Image.open(path) as img:
                    ratio = detect_and_calculate_area_ratio(img)

                    # Check the condition and append to list if it matches
                    # This should be somewhere between 0.4 and 0.5
                    if ratio is not None and (ratio < 0.75):
                        target_images.append(path)

    return png_paths, target_images


image_path = 'data/rbc_images'
png_paths, target_images = gather_images(image_path)


def get_image_descriptors(image_paths):
    # Initialize an empty list to store each image's descriptor dictionary
    all_descriptors = []

    for image_path in image_paths:
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

        # If there are no regions, continue to the next image
        if not regions:
            continue

        # Select the region with the largest area
        region = max(regions, key=lambda region: region.area)

        descriptor_dict = {
            'area': region.area,
            'filled_area': region.filled_area,
            'equivalent_diameter': region.equivalent_diameter,
            'eccentricity': region.eccentricity,
            'convex_area': region.convex_area,
            'extent': region.extent,
            'solidity': region.solidity,
            'perimeter': region.perimeter,
            'image_path': image_path,
        }

        all_descriptors.append(descriptor_dict)

    # Convert the list of dictionaries into a DataFrame
    df_1 = pd.DataFrame(all_descriptors)

    df = pd.DataFrame(all_descriptors)

    # Add an extra column 'Source' for identification
    df['Source'] = df['image_path'].str.contains('sma')

    # Reshape the dataframe suitable for sns.boxplot
    df_melt = df.melt(id_vars=['Source', 'image_path'])

    # Get the unique column names (variables)
    columns = df_melt['variable'].unique()

    return df_1


print(f'Cleaning images...')


df = get_image_descriptors(png_paths)

# Filter the DataFrame for small regions
small_regions_df_count = df[df['area'] < 5000].count()

# Filter the DataFrame
small_regions_df = df[df['area'] < 5000]

# Get the unique image paths
small_regions_paths = small_regions_df['image_path'].unique()

small_regions_paths_list = list(small_regions_df['image_path'].unique())

# Filter the DataFrame for small regions
large_regions_df_count = df[df['area'] > 13000].count()

# Filter the DataFrame
large_regions_df = df[df['area'] > 13000]

# Get the unique image paths
large_regions_paths = large_regions_df['image_path'].unique()

large_regions_paths_list = list(large_regions_df['image_path'].unique())

png_paths = list(
    filter(lambda path: path not in small_regions_paths_list, png_paths),
)
png_paths = list(
    filter(lambda path: path not in large_regions_paths_list, png_paths),
)

# Filter the DataFrame
p_large_regions_df = df[df['perimeter'] > 750]

# Get the unique image paths
p_large_regions_paths = p_large_regions_df['image_path'].unique()

p_large_regions_paths_list = list(p_large_regions_df['image_path'].unique())

png_paths = list(
    filter(lambda path: path not in p_large_regions_paths_list, png_paths),
)
len(png_paths)

df = get_image_descriptors(png_paths)

# Concatenate the lists
all_paths = target_images + p_large_regions_paths_list + \
    large_regions_paths_list + small_regions_paths_list
# print(len(all_paths))

# Create a DataFrame with all paths under column 'Paths'
df = pd.DataFrame(all_paths, columns=['Paths'])
df = df.drop_duplicates()

# Define your source and destination directories
src_dir = 'data/rbc_images'
dest_dir = 'data/rbc_images_cleaned'

# Ensure destination directory exists
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Copy all files from src_dir to dest_dir while maintaining the directory structure
for root, dirs, files in os.walk(src_dir):
    # Determine the path to the destination directory
    dest_path = root.replace(src_dir, dest_dir, 1)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for file in files:
        src_file_path = os.path.join(root, file)
        dest_file_path = os.path.join(dest_path, file)
        shutil.copy(src_file_path, dest_file_path)

deleted_files_count = 0


# Delete specific files from the destination directory
for file_path in df['Paths']:
    # Skip if file_path is NaN or not a string
    if pd.isna(file_path) or not isinstance(file_path, str):
        continue

    # Adjust the path to the new directory structure
    new_path = file_path.replace(
        'data/rbc_images', 'data/rbc_images_cleaned', 1,
    )
    try:
        os.remove(new_path)
        # print(f"Deleted: {new_path}")
        deleted_files_count += 1  # Increment the counter

    except FileNotFoundError:
        print(f'File not found: {new_path}')
    except Exception as e:
        print(f'Error deleting {new_path}: {e}')

print(
    f'Cleaning process is completed. Total files deleted: {deleted_files_count}.',
)
