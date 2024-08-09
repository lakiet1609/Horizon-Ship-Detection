from pydantic_settings import BaseSettings

class GeneralSettings(BaseSettings):
    input_path: str = 'data/three_ships_horizon.JPG'
    output_path: str = 'results'

class HorizontalDetectionSettings(BaseSettings):
    file_name: str = 'horizontal_detection.png'

class RotateSettings(BaseSettings):
    output_folder: str = 'results/rotate'
    file_name: str = 'rotate.png'

class CropSettings(BaseSettings):
    file_name: str = 'crop.png'

class ShipDetectionSettings(BaseSettings):
    gpu: str = '0'
    edge_output_path: str = 'edge_detection'
    checkpoint: str = 'models/bsds500_pascal_model.pth'
    file_name: str = 'output_image.tiff'
    mask_file_name: str = 'mask_bbox.png'
    start_point: tuple = (0, 205)
    end_point: tuple = (1600, 203)
    thickness: int = 5
    erode_kernel: tuple = (3,3)
    erode_iterations: int = 1
    dilate_kernel: tuple = (3,3)
    dilate_iterations: int = 6
    min_width: int = 32 
    min_height: int = 32 
    expansion_size:int = 50

general_settings = GeneralSettings()
horizontal_settings = HorizontalDetectionSettings()
rotate_settings = RotateSettings()
crop_settings = CropSettings()
ship_detection_settings = ShipDetectionSettings()

