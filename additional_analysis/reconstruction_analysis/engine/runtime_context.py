"""
config.py
Static defaults and physical constants.
Derived calculations (DX, DY) removed to support dynamic command line args.
"""
from logger import log_message, log_image

# Default Grid Dimensions
NX_DEFAULT = 128
NY_DEFAULT = 96

# Physical Dimensions (m)
PHYSICAL_SIZE_X = 0.228447 #0.13 #0.2
PHYSICAL_SIZE_Y = 0.228186 #0.08 #0.15

# Speed of Sound (SoS) Values (m/s)
SOS_WATER = 1500.0
SOS_MIN = 1400.0
SOS_MAX = 1650.0

# Hardware Defaults
N_EMITTERS_DEFAULT = 32
N_RECEIVERS_DEFAULT = 32
RADIUS_DEFAULT = 0.1091

# Training Defaults
NUM_SAMPLES_DEFAULT = 1000
MAX_SHAPES = 8
MIN_SHAPES = 4
BATCH_SIZE_DEFAULT = 32
LEARNING_RATE_DEFAULT = 1e-3
EPOCHS_DEFAULT = 50
DEVICE_DEFAULT = "cpu"










