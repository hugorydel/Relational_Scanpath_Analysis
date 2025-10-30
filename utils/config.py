# Data paths
DATA_ROOT = "D:"  # External drive location
COCO_PATH = f"{DATA_ROOT}/Relational Scanpath Research/coco_data"  # Adjust based on your structure


# Helper function for accessing config values
def get(key, default=None):
    """Get configuration value with default fallback"""
    return globals().get(key, default)
