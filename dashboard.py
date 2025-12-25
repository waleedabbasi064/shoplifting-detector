import streamlit as st
import os
import pandas as pd
import time
from PIL import Image

# Configuration
EVIDENCE_FOLDER = 'evidence'
st.set_page_config(page_title="Security Control Room", layout="wide")

# Custom CSS for "Security" feel
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚨 Retail Security: Shoplifting Monitor")
st.markdown("---")

# Placeholder for real-time metrics
metrics_placeholder = st.empty()
gallery_placeholder = st.empty()

def load_evidence():
    if not os.path.exists(EVIDENCE_FOLDER):
        return []
    files = [f for f in os.listdir(EVIDENCE_FOLDER) if f.endswith('.jpg') or f.endswith('.png')]
    # Sort by time (newest first) - assuming filename has timestamp or using file metadata
    files.sort(key=lambda x: os.path.getmtime(os.path.join(EVIDENCE_FOLDER, x)), reverse=True)
    return files

def get_stats(files):
    total_alerts = len(files)
    if total_alerts > 0:
        last_alert = time.ctime(os.path.getmtime(os.path.join(EVIDENCE_FOLDER, files[0])))
    else:
        last_alert = "No alerts yet"
    return total_alerts, last_alert

# Auto-refresh loop
while True:
    evidence_files = load_evidence()
    total_alerts, last_alert_time = get_stats(evidence_files)

    # 1. Update Metrics
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Total Suspicious Incidents", value=total_alerts)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="System Status", value="ONLINE 🟢")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Last Detection", value=last_alert_time)
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. Update Gallery (Show top 8 recent images)
    with gallery_placeholder.container():
        st.subheader("📸 Recent Evidence Log")
        if not evidence_files:
            st.info("No suspicious activity detected yet. System is scanning...")
        else:
            # Create grid layout
            cols = st.columns(4)
            for idx, file_name in enumerate(evidence_files[:8]): # Show max 8
                file_path = os.path.join(EVIDENCE_FOLDER, file_name)
                image = Image.open(file_path)
                with cols[idx % 4]:
                    st.image(image, caption=file_name, use_container_width=True)
                    st.error(f"Alert #{len(evidence_files) - idx}")

    # Refresh every 2 seconds
    time.sleep(2)
    st.rerun()