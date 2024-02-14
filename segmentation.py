from __future__ import annotations

from src.rbc_segmentation.utils import ensure_directories_exist
from src.rbc_segmentation.utils import rbc_segm_folders

ensure_directories_exist()

rbc_segm_folders(
    input_relative_path='data/whole_slide_images/non-sma',
    output_relative_path='data/rbc_images/non-sma',
)

rbc_segm_folders(
    input_relative_path='data/whole_slide_images/sma',
    output_relative_path='data/rbc_images/sma',
)
