# Cat Litter Box Monitor

A Python application that monitors multiple litter boxes using IP cameras and sends Discord notifications when cats spend too long in the box, potentially indicating health issues.

## Features

- Multi-camera support for monitoring multiple litter boxes
- Real-time cat detection using YOLOv8
- Video clip recording of litter box visits
- Discord notifications for extended litter box usage
- Automatic camera reconnection on connection issues
- Configurable thresholds and camera sources

## Requirements

- Python 3.8+ (I used 3.11)
- OpenCV
- YOLOv8
- IP Cameras with RTSP support (I used TPLink Tapo C200s)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cat-litter-monitor.git
   cd cat-litter-monitor
   ```


2. Configure the application:
   - Create a `.env` file with the following variables:
     - `DISCORD_WEBHOOK_URL`
     - `CAMERA_SOURCE[1-n]` (should include authentication in the RTSP URL if needed)
     - etc.

4. Build the Docker image and run a container
   ```bash
   docker build -t cat-monitor .
   docker run --env-file .env cat-monitor
   ```

