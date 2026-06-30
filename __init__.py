"""
CloudKiller - Generative AI-Based Cloud Removal for LISS-IV Satellite Imagery
Project Structure Initialization
"""

__version__ = "1.0.0"
__author__ = "CloudKiller Team"
__description__ = "Cloud removal and surface reconstruction for LISS-IV satellite imagery"

import sys
from pathlib import Path

# Add root to path for imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print(f"CloudKiller initialized from: {ROOT_DIR}")