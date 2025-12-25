🚨 AI-Powered Shoplifting Detection System

An automated security system that detects suspicious behavior (concealment) in retail environments using YOLOv12, Pose Estimation, and Real-Time Logic.

🎯 Features

Object Tracking: Uses YOLOv12 + ByteTrack to track customers.

Pose Estimation: Detects hand movements relative to body pockets.

Behavior Analysis: Identifies "concealment" logic (putting items in pockets) inside restricted zones.

Control Room Dashboard: A real-time web interface (Streamlit) for security guards.

Automated Alerts: Captures evidence photos instantly upon detection.

🛠️ Tech Stack

Python 3.10+

YOLOv12 (Object Detection)

YOLOv8-Pose (Keypoint Detection)

Streamlit (Web Dashboard)

OpenCV (Video Processing)

�� Installation

Clone the repository

git clone [https://github.com/YOUR_USERNAME/shoplifting-detection.git](https://github.com/YOUR_USERNAME/shoplifting-detection.git)
cd shoplifting-detection


Install dependencies

pip install -r requirements.txt


Setup the Environment
(Optional) If you have a specific video file, place it in the root folder and name it 1.mp4.

💻 Usage

1. Run the AI Engine
This script processes the video and detects theft.

python main_final.py


2. Launch the Dashboard
Open a new terminal to view live alerts.

streamlit run dashboard.py


📂 Project Structure

main_final.py: The core AI engine.

dashboard.py: The security control room interface.

shoplifting_logic.py: Helper functions for behavior logic.

my_model.pt: The fine-tuned YOLOv12 model.

*(Save and exit)*
