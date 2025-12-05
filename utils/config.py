# Data paths
DATA_ROOT = "D:"  # External drive location
COCO_PATH = f"{DATA_ROOT}/Relational Scanpath Research/coco_data"  # Adjust based on your structure

# Category exclusions - images containing ANY of these categories will be filtered out
EXCLUDED_CATEGORIES = [
    "pizza",
]


# Helper function for accessing config values
def get(key, default=None):
    """Get configuration value with default fallback"""
    return globals().get(key, default)
