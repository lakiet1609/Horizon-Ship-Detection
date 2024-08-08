from pydantic_settings import BaseSettings

class GeneralSettings(BaseSettings):
    input_path: str = 'data'
    output_path: str = 'results'


class ShipDetectionSettings(BaseSettings):
    gpu: int = 0
    output_path: str = 'results/edge_detection'
    checkpoint: str = 'models/bsds500_pascal_model.pth'


general_settings = GeneralSettings()
ship_detection_settings = ShipDetectionSettings()

