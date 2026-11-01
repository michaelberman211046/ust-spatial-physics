"""
config.py
This ensures that when any script imports config, the src/ directory is added to the Python import path.
"""
import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH
_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))









