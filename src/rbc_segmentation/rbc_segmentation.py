

import argparse
import os
import argparse 
from utils import rbc_segm_folders
import os

rbc_segm_folders(
    input_relative_path = 'data/whole_slide_images/non-sma',
    output_relative_path = 'data/rbc_images/non-sma'
    )

rbc_segm_folders(
    input_relative_path = 'data/whole_slide_images/sma',
    output_relative_path = 'data/rbc_images/sma'
    )