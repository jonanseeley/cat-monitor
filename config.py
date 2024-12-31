from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configuration with defaults
THRESHOLD = int(os.getenv('LITTER_BOX_THRESHOLD', 45))
CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', 0)
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)