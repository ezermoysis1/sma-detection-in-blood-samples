import os
import cv2
import numpy as np
import imageio
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from tqdm import tqdm

def crop_and_save3(complete_image, centroid, shapeid, output_dir, size=160):
    """ Crops and saves a region from an image """
    nx_0 = max(int(centroid[0] - size/2),0)
    ny_0 = max(int(centroid[1] - size/2),0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    roi_file = os.path.join(output_dir, str(shapeid)+'.png')
    cropped_image = complete_image[ny_0:ny_1, nx_0: nx_1,:]
    imageio.imwrite(roi_file, cropped_image)    
    
def rbc_segmentation(img, outputdir=None):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=2)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    bw=(markers>1).astype(int)
    bw=binary_fill_holes(bw).astype(np.uint8)
    bw_clean=morphology.remove_small_objects(bw.astype(bool), min_size=5000, connectivity=4).astype(np.uint8)
    rbc_gone = 255*remove_small_objects(bw_clean.astype(bool), min_size=17000, connectivity=4).astype(np.uint8)
    rbc_only = cv2.subtract(255*bw_clean, rbc_gone)
    return rbc_only


def chop_thumbnails(image, output_dir, current_shapeid=0):
    shapeid = current_shapeid
    mp_masks=rbc_segmentation(image)
    output  = cv2.connectedComponentsWithStats(mp_masks, connectivity=8)
    centroids = output[3]    
    for c in centroids:
        crop_and_save3(image, c, shapeid, output_dir)
        shapeid=shapeid+1        
    return shapeid

# Function for RBC segmentation and thumbnail extraction
def process_image(image, output_dir):
    # Perform RBC segmentation
    rbc_only = rbc_segmentation(image)
    print(rbc_only)
    # Perform thumbnail extraction
    shapeid = chop_thumbnails(image, output_dir)
    
    return shapeid

def rbc_segm_folders(input_relative_path, output_relative_path):

    # Define input and output directories
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, input_relative_path)
    output_dir = os.path.join(current_dir, output_relative_path)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of input folders
    input_folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

    # Process each folder
    for folder in tqdm(input_folders, desc='Processing folders'):
        # Create corresponding folder in the output directory
        output_folder = os.path.join(output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Get a list of image files in the current folder
        folder_path = os.path.join(input_dir, folder)
        image_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff')]
        
        # Process each image in the current folder
        for image_file in tqdm(image_files, desc=f'Processing images in folder {folder}'):
            # Read the image
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Process the image
                shapeid = process_image(image, output_folder)
                
                print(f"Image {image_file} processed. Total shapes extracted: {shapeid}")

    print("All images processed successfully.")