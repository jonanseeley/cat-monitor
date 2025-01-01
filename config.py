from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configuration with defaults
THRESHOLD = int(os.getenv('LITTER_BOX_THRESHOLD', 45))
CAMERA_SOURCE1 = os.getenv('CAMERA_SOURCE1', 0)
CAMERA_SOURCE2 = os.getenv('CAMERA_SOURCE2', 0)
CAMERA_SOURCE3 = os.getenv('CAMERA_SOURCE3', 0)
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
if CAMERA_SOURCE1.isdigit():
    CAMERA_SOURCE1 = int(CAMERA_SOURCE1)
if CAMERA_SOURCE2.isdigit():
    CAMERA_SOURCE2 = int(CAMERA_SOURCE2)
if CAMERA_SOURCE3.isdigit():
    CAMERA_SOURCE3 = int(CAMERA_SOURCE3)