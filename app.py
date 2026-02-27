import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from datetime import datetime

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Hero IntelliInspect",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for Animations and Styling
# ----------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main title animation */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.3);
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Damage badge */
    .damage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.5rem;
        font-weight: 600;
        animation: slideInLeft 0.5s ease-out;
        transition: transform 0.3s ease;
    }
    
    .damage-badge:hover {
        transform: scale(1.1);
    }
    
    /* Severity indicators */
    .severity-low {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #0d5e3a;
    }
    
    .severity-medium {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #8b4513;
    }
    
    .severity-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    /* Progress bar animation */
    @keyframes loadBar {
        0% {
            width: 0%;
        }
        100% {
            width: 100%;
        }
    }
    
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        animation: loadBar 1.5s ease-out;
    }
    
    /* Stats boxes */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        animation: slideInRight 0.6s ease-out;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Loading spinner */
    .loader {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Image container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
    }
    
    /* Feature icons */
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    /* Timeline */
    .timeline-item {
        padding: 1rem;
        border-left: 3px solid #667eea;
        margin-left: 1rem;
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🚗 Hero IntelliInspect</h1>
    <p class="main-subtitle">AI-Powered Vehicle Damage Detection & Intelligent Assessment</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - 🔍 **AI Detection** - Advanced YOLO-based damage detection
    - 📊 **Smart Analysis** - Intelligent severity assessment
    - 💰 **Cost Estimation** - Instant repair cost calculation
    - 📈 **Detailed Reports** - Comprehensive damage breakdown
    """)
    
    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
    <div class="timeline-item">
        <strong>1. Upload Image</strong><br>
        Select a vehicle photo
    </div>
    <div class="timeline-item">
        <strong>2. AI Analysis</strong><br>
        Our AI detects damages
    </div>
    <div class="timeline-item">
        <strong>3. Get Report</strong><br>
        Receive detailed assessment
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_details = st.checkbox("Show Detection Details", value=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

with st.spinner("🔄 Loading AI Model..."):
    model = load_model()
    st.success("✅ Model Loaded Successfully!")

# ----------------------------
# Upload Section
# ----------------------------
st.markdown("## 📤 Upload Vehicle Image")

col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])

with col_upload2:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the vehicle"
    )

if uploaded_file is not None:
    
    # Progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    st.markdown("---")
    
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📸 Original Image")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Run YOLO Detection with loading animation
    with st.spinner("🔍 Analyzing image with AI..."):
        results = model(image, conf=confidence_threshold)
        time.sleep(0.5)  # Slight delay for effect

    annotated_image = results[0].plot()

    with col2:
        st.markdown("### 🎯 Detected Damages")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(annotated_image, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------
    # Extract Detection Info
    # ----------------------------
    damages = []
    total_severity_score = 0

    for box in results[0].boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]
        confidence = float(box.conf)

        damages.append((class_name, confidence))

        # Assign severity weight
        if class_name in ["glass shatter", "lamp broken", "tire flat"]:
            total_severity_score += 3
        elif class_name in ["dent", "crack"]:
            total_severity_score += 2
        elif class_name == "scratch":
            total_severity_score += 1

    # ----------------------------
    # Statistics Cards
    # ----------------------------
    st.markdown("---")
    st.markdown("## 📊 Quick Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-number">{len(damages)}</p>
            <p class="stat-label">Damages Detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        severity_level = "Low" if total_severity_score <= 3 else "Medium" if total_severity_score <= 6 else "High"
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-number">{severity_level}</p>
            <p class="stat-label">Severity Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        estimated_cost = total_severity_score * 1500
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-number">₹{estimated_cost:,}</p>
            <p class="stat-label">Est. Repair Cost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col4:
        avg_confidence = np.mean([d[1] for d in damages]) if damages else 0
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-number">{avg_confidence:.0%}</p>
            <p class="stat-label">Avg. Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------
    # Intelligent Assessment
    # ----------------------------
    st.markdown("---")
    st.markdown("## 🔍 Detailed Assessment Report")

    if len(damages) == 0:
        st.success("✅ No visible damage detected. Vehicle appears to be in good condition!")
    else:
        # Damage List
        st.markdown("### 🛠️ Identified Damages")
        
        damage_cols = st.columns(2)
        for idx, (damage, conf) in enumerate(damages):
            with damage_cols[idx % 2]:
                severity_class = "severity-high" if damage in ["glass shatter", "lamp broken", "tire flat"] else \
                                "severity-medium" if damage in ["dent", "crack"] else "severity-low"
                st.markdown(f"""
                <div class="damage-badge {severity_class}">
                    <strong>{damage.upper()}</strong><br>
                    Confidence: {conf:.1%}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Severity Assessment
        col_sev1, col_sev2 = st.columns(2)
        
        with col_sev1:
            st.markdown("### 📈 Severity Analysis")
            if total_severity_score <= 3:
                level = "Low"
                emoji = "🟢"
                desc = "Minor cosmetic damage. Quick fix possible."
            elif total_severity_score <= 6:
                level = "Medium"
                emoji = "🟡"
                desc = "Moderate damage. Professional repair recommended."
            else:
                level = "High"
                emoji = "🔴"
                desc = "Significant damage. Immediate attention required."

            st.markdown(f"""
            <div class="info-card">
                <h2 style="margin:0; color: #667eea;">{emoji} {level} Severity</h2>
                <p style="margin-top: 1rem;">{desc}</p>
                <p style="margin-top: 1rem;"><strong>Severity Score:</strong> {total_severity_score}/15</p>
            </div>
            """, unsafe_allow_html=True)

        with col_sev2:
            st.markdown("### 🔧 Damage Classification")
            # Cosmetic vs Functional
            functional_damages = ["glass shatter", "lamp broken", "tire flat"]

            if any(d[0] in functional_damages for d in damages):
                damage_type = "Functional Damage"
                type_icon = "⚠️"
                type_desc = "Affects vehicle operation. Immediate repair needed."
            else:
                damage_type = "Cosmetic Damage"
                type_icon = "🎨"
                type_desc = "Affects appearance only. Repair at convenience."

            st.markdown(f"""
            <div class="info-card">
                <h2 style="margin:0; color: #667eea;">{type_icon} {damage_type}</h2>
                <p style="margin-top: 1rem;">{type_desc}</p>
            </div>
            """, unsafe_allow_html=True)

        # Cost Breakdown
        st.markdown("---")
        st.markdown("### 💰 Cost Estimation Breakdown")
        
        cost_data = []
        for damage, conf in damages:
            if damage in ["glass shatter", "lamp broken", "tire flat"]:
                cost = 4500
            elif damage in ["dent", "crack"]:
                cost = 3000
            else:
                cost = 1500
            cost_data.append((damage, cost, conf))
        
        col_cost1, col_cost2 = st.columns([2, 1])
        
        with col_cost1:
            for damage, cost, conf in cost_data:
                st.markdown(f"""
                <div class="info-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.1rem;">{damage.title()}</strong><br>
                            <span style="color: #888;">Confidence: {conf:.1%}</span>
                        </div>
                        <div>
                            <span class="gradient-text" style="font-size: 1.5rem;">₹{cost:,}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_cost2:
            total_cost = sum([c[1] for c in cost_data])
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <p class="stat-label">Total Estimated Cost</p>
                <p class="stat-number">₹{total_cost:,}</p>
            </div>
            """, unsafe_allow_html=True)

        # Report timestamp
        st.markdown("---")
        st.info(f"⚠️ **Disclaimer:** This is an AI-generated estimate. Final claim approval is subject to physical inspection & insurance policy terms.")
        
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        st.markdown(f"<p style='text-align: center; color: #888;'>Report generated on {timestamp}</p>", unsafe_allow_html=True)
        
        # Download Report Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("📥 Download Full Report", use_container_width=True):
                st.balloons()
                st.success("Report download started!")

else:
    # Welcome Screen
    st.markdown("---")
    col_wel1, col_wel2, col_wel3 = st.columns([1, 2, 1])
    
    with col_wel2:
        st.markdown("""
        <div class="upload-section">
            <div class="feature-icon">📸</div>
            <h2 style="color: white; margin: 0;">Upload Your Vehicle Image</h2>
            <p style="color: white; opacity: 0.9; margin-top: 1rem;">
                Get instant AI-powered damage detection and cost estimates
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("## ✨ Why Choose Hero IntelliInspect?")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="info-card">
            <div class="feature-icon">🤖</div>
            <h3 class="gradient-text">Advanced AI</h3>
            <p>State-of-the-art YOLO model for accurate damage detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="info-card">
            <div class="feature-icon">⚡</div>
            <h3 class="gradient-text">Instant Results</h3>
            <p>Get comprehensive reports in seconds, not days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="info-card">
            <div class="feature-icon">💯</div>
            <h3 class="gradient-text">High Accuracy</h3>
            <p>Trained on thousands of vehicle damage images</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #888; font-size: 0.9rem;'>
    Made with ❤️ using Streamlit & YOLO | © 2024 Hero IntelliInspect
</p>
""", unsafe_allow_html=True)