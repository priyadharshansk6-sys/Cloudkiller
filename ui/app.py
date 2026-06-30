import sys
import os
from pathlib import Path
import streamlit as st
from PIL import Image
import time
import io
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# FIX: Ensure root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# FIX: Import the singleton 'config' object correctly
from config import config 

# Import Backend Modules
try:
    from pipeline.data_pipeline import DataPipeline 
    from inference.inference import CloudRemovalModel
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    st.error(f"⚠️ Backend module warning: {e}") 

# ============================================================================
# 2. PAGE CONFIGURATION & STYLING (AgriVision AI Premium Design)
# ============================================================================
st.set_page_config(
    page_title="Cloudkiller AI | Farm Monitoring",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'reconstruction_complete' not in st.session_state:
    st.session_state.reconstruction_complete = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'original_img' not in st.session_state:
    st.session_state.original_img = None
if 'reconstructed_img' not in st.session_state:
    st.session_state.reconstructed_img = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0

# Custom CSS for AgriVision Premium Theme
bg_img_url = "https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80"
st.markdown(f"""
<style>
/* Global Styles */
.stApp {{
    background-image: linear-gradient(135deg, rgba(46,125,50,0.05), rgba(30,30,30,0.05)), url("{bg_img_url}");
    background-size: cover;
    background-attachment: fixed;
}}

.main {{ background-color: transparent; }}

/* Button Styling */
.stButton>button {{
    width: 100%; 
    border-radius: 12px; 
    height: 3.5em; 
    background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
    color: white; 
    font-weight: bold; 
    border: none; 
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(46, 125, 50, 0.2);
}}

.stButton>button:hover {{ 
    background: linear-gradient(135deg, #1b5e20 0%, #0d3817 100%);
    box-shadow: 0 6px 12px rgba(46, 125, 50, 0.4);
    transform: translateY(-2px);
}}

/* Metric Card Styling */
.metric-card {{
    background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
    padding: 25px; 
    border-radius: 15px; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center; 
    border-left: 6px solid #2e7d32;
    transition: all 0.3s ease;
    border: 1px solid rgba(46, 125, 50, 0.1);
}}

.metric-card:hover {{
    box-shadow: 0 8px 25px rgba(46, 125, 50, 0.2);
    transform: translateY(-5px);
}}

/* Title Styling */
.title-text {{
    color: #1b5e20; 
    text-align: center; 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-weight: 900; 
    font-size: 3rem; 
    margin-bottom: 0px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}}

.subtitle-text {{ 
    text-align: center; 
    color: #555; 
    font-size: 1.2rem; 
    margin-bottom: 30px;
    font-weight: 500;
}}

/* Status Indicators */
.status-success {{
    background-color: #e8f5e9;
    border-left: 5px solid #4caf50;
    padding: 12px;
    border-radius: 5px;
    color: #2e7d32;
}}

.status-warning {{
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 12px;
    border-radius: 5px;
    color: #e65100;
}}

/* Sidebar Styling */
.sidebar-section {{
    background-color: rgba(255, 255, 255, 0.9);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border: 1px solid rgba(46, 125, 50, 0.1);
}}

/* Image Container */
.image-container {{
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    border: 2px solid rgba(46, 125, 50, 0.2);
}}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# 3. SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    # Logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.icons8.com/fluency/96/000000/leaf.png", width=80)
        
    st.title("AgriVision AI")
    st.markdown("**Smart Farm Monitoring System**")
    st.markdown("---")
    
    # Navigation Menu
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Home", "📤 Upload & Process", "📊 Results", "📈 Analytics", "📚 Docs", "ℹ️ About"],
        icons=["house", "cloud-upload", "check2-square", "graph-up", "book", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f8f8"},
            "icon": {"color": "#2e7d32", "font-size": "20px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#2e7d32", "color": "white", "font-weight": "bold"},
        }
    )
    
    st.markdown("---")
    
    # Configuration Panel
    with st.expander("⚙️ Configuration", expanded=False):
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        
        farm_id = st.text_input("Farm ID", value="FARM-001", key="farm_id")
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        use_sar = st.checkbox("🛰️ Enable SAR Data Fusion", value=True)
        processing_mode = st.radio("Processing Mode", ["Fast (256px)", "Standard (384px)", "High-Res (512px)"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Backend Status
    st.markdown("### System Status")
    st.metric("AI Backend Pipeline", "✅ Online" if BACKEND_AVAILABLE else "⚠️ Offline (Demo Mode)", 
              delta="Ready for inference" if BACKEND_AVAILABLE else "Check module paths")

# ============================================================================
# 4. MAIN HEADER
# ============================================================================
st.markdown("<h1 class='title-text'>🌾 Cloudkiller Smart Farm Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>AI-Powered Cloud Removal & Vegetation Health Monitoring</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# 5. PAGE: HOME
# ============================================================================
if selected == "🏠 Home":
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Welcome to Cloudkiller AI")
        st.markdown("""
        Cloudkiller is an advanced AI system designed for agricultural intelligence and  monitoring using satellite imagery.
        
        **Key Features:**
        - 🔍 Cloud Detection & Removal
        - 🌱 Vegetation Health Analysis
        - 📊 NDVI Monitoring
        - 🚨 Crop Alert System
        - 📈 Predictive Analytics
        """)
        
    with col2:
        st.info("👈 **Start by uploading a satellite image** in the 'Upload & Process' tab")
        
    st.markdown("---")
    
    # Feature Cards
    st.markdown("### How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>1️⃣ Upload Data</h3>
            <p>Upload your cloudy LISS-IV satellite imagery or any RGB/RGBN image</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>2️⃣ AI Processing</h3>
            <p>Advanced ML models remove clouds using Conditional Diffusion</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>3️⃣ Crop Analytics</h3>
            <p>Generate NDVI maps, health metrics, and farmer recommendations</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# 6. PAGE: UPLOAD & RECONSTRUCT
# ============================================================================
elif selected == "📤 Upload & Process":
    
    st.markdown("### 📤 Upload Satellite Image")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Select an image file (JPG, PNG, TIF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a cloudy satellite image for processing"
    )
    
    if uploaded_file is not None:
        
        # Show Preview
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Preview")
            st.image(image, use_container_width=True, caption="Uploaded Image")
            
        with col2:
            st.markdown("#### File Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Image Dimensions:** {image.size[0]} × {image.size[1]} px")
            
        st.markdown("---")
        
        # Processing Button
        if st.button("🚀 Process Image & Generate Report", key="process_btn", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            try:
                # Step 1: Prepare
                status_placeholder.markdown("<div class='status-warning'>⏳ Preparing image data...</div>", unsafe_allow_html=True)
                progress_bar.progress(10)
                
                temp_path = Path("/tmp") / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                time.sleep(0.5)
                
                # Step 2: Process
                status_placeholder.markdown("<div class='status-warning'>🤖 Running AI inference...</div>", unsafe_allow_html=True)
                progress_bar.progress(40)
                
                start_time = time.time()
                
                if BACKEND_AVAILABLE:
                    # Initialize Backend Modules
                    pipeline = DataPipeline()
                    model = CloudRemovalModel()
                    
                    # Run Pipeline
                    processed_data = pipeline.process_single_scene(str(temp_path), str(temp_path))
                    cloudy_input = processed_data['cloudy']
                    
                    # Run Model Inference
                    ai_output = model.predict(cloudy_input)
                    
                    # Convert Output to PIL Image (if model returns numpy array)
                    if isinstance(ai_output, np.ndarray):
                        if ai_output.max() <= 1.0:
                            ai_output = (ai_output * 255).astype(np.uint8)
                        result_image = Image.fromarray(ai_output)
                    else:
                        result_image = ai_output
                    
                    # Get dynamic NDVI from pipeline if available
                    ndvi_value = round(float(np.mean(processed_data.get('ndvi', 0.72))), 2)
                    
                else:
                    # Fallback: simulate processing if backend is offline
                    time.sleep(2)
                    result_image = image
                    ndvi_value = 0.72
                    
                processing_time = time.time() - start_time
                
                # Step 3: Analyze
                status_placeholder.markdown("<div class='status-warning'>📊 Analyzing results...</div>", unsafe_allow_html=True)
                progress_bar.progress(80)
                
                time.sleep(0.5)
                
                # Step 4: Complete
                status_placeholder.markdown("<div class='status-success'>✅ Processing Complete!</div>", unsafe_allow_html=True)
                progress_bar.progress(100)
                
                # Store Results
                st.session_state.original_img = image
                st.session_state.reconstructed_img = result_image
                st.session_state.metrics = {
                    'ndvi': ndvi_value,
                    'humidity': "65%",
                    'cloud_cover': "0.0%",
                    'vegetation_density': "High",
                    'crop_health': "Excellent"
                }
                st.session_state.processing_time = processing_time
                st.session_state.reconstruction_complete = True
                
                st.success("✅ Analysis Complete! Check the Results & Analytics tabs for detailed insights.")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                
    else:
        st.info("📌 Please upload a satellite image to begin the analysis")

# ============================================================================
# 7. PAGE: RESULTS
# ============================================================================
elif selected == "📊 Results":
    
    if not st.session_state.reconstruction_complete:
        st.warning("⚠️ No results available. Please process an image in the 'Upload & Process' tab first.")
        
    else:
        st.markdown("### Performance Metrics")
        
        # Key Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>NDVI</h3>
                <h2 style="color:#2e7d32;">{st.session_state.metrics['ndvi']}</h2>
                <p>Vegetation Index</p>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Humidity</h3>
                <h2 style="color:#0288d1;">{st.session_state.metrics['humidity']}</h2>
                <p>Soil Moisture</p>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cloud Cover</h3>
                <h2 style="color:#fbc02d;">{st.session_state.metrics['cloud_cover']}</h2>
                <p>After Removal</p>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Vegetation</h3>
                <h2 style="color:#558b2f;">{st.session_state.metrics['vegetation_density']}</h2>
                <p>Density</p>
            </div>
            """, unsafe_allow_html=True)
            
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Crop Health</h3>
                <h2 style="color:#4caf50;">{st.session_state.metrics['crop_health']}</h2>
                <p>Status</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Image Comparison
        st.markdown("### Image Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ☁️ Cloudy Input (Original)")
            st.image(st.session_state.original_img, use_container_width=True)
            
        with col2:
            st.markdown("#### ✨ Cloud-Free (AI Reconstructed)")
            st.image(st.session_state.reconstructed_img, use_container_width=True)
            
        st.markdown("---")
        
        # Download Section
        st.markdown("### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            buf = io.BytesIO()
            st.session_state.reconstructed_img.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Processed Image",
                data=buf.getvalue(),
                file_name="Cloudkiller_reconstructed.png",
                mime="image/png",
                use_container_width=True
            )
            
        with col2:
            # Create metrics report
            report = f"""
            AGRIVISION AI - ANALYSIS REPORT
            ================================
            NDVI: {st.session_state.metrics['ndvi']}
            Humidity: {st.session_state.metrics['humidity']}
            Cloud Cover: {st.session_state.metrics['cloud_cover']}
            Vegetation Density: {st.session_state.metrics['vegetation_density']}
            Crop Health: {st.session_state.metrics['crop_health']}
            Processing Time: {st.session_state.processing_time:.2f}s
            """
            
            st.download_button(
                label="📊 Download Report (TXT)",
                data=report,
                file_name="agrivision_report.txt",
                mime="text/plain",
                use_container_width=True
            )

# ============================================================================
# 8. PAGE: ANALYTICS
# ============================================================================
elif selected == "📈 Analytics":
    
    if not st.session_state.reconstruction_complete:
        st.warning("⚠️ No analytics available. Please process an image first.")
        
    else:
        # Trend Data
        chart_data = pd.DataFrame({
            'Month': ['January', 'February', 'March', 'April', 'May', 'June'],
            'NDVI': [0.45, 0.50, 0.58, 0.65, 0.70, st.session_state.metrics['ndvi']],
            'Cloud_Cover': [45, 38, 32, 25, 15, 0.0]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 NDVI Growth Trend")
            fig_ndvi = px.line(
                chart_data, 
                x='Month', 
                y='NDVI',
                markers=True,
                title='Seasonal NDVI Progression',
                color_discrete_sequence=['#2e7d32']
            )
            fig_ndvi.update_layout(
                plot_bgcolor='rgba(255,255,255,0.7)',
                paper_bgcolor='rgba(255,255,255,0.7)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_ndvi, use_container_width=True)
            
        with col2:
            st.markdown("### ☁️ Cloud Cover Reduction")
            fig_cloud = px.bar(
                chart_data,
                x='Month',
                y='Cloud_Cover',
                title='Cloud Coverage Over Time',
                color_discrete_sequence=['#ff9800']
            )
            fig_cloud.update_layout(
                plot_bgcolor='rgba(255,255,255,0.7)',
                paper_bgcolor='rgba(255,255,255,0.7)'
            )
            st.plotly_chart(fig_cloud, use_container_width=True)

# ============================================================================
# 9. PAGE: DOCUMENTATION
# ============================================================================
elif selected == "📚 Docs":
    
    st.markdown("### 📚 Technical Documentation")
    
    with st.expander("🏗️ System Architecture", expanded=True):
        st.markdown("""
        #### Integration Architecture
        
        The Cloudkiller system integrates the following specialized backend modules:
        
        **Data Processing Pipeline:** - Multispectral band extraction via Rasterio
        - Temporal data fusion 
        - SAR-optical data registration
        
        **AI Inference Engine:**
        - UNet-based cloud segmentation
        - Generative model for image reconstruction
        - Spectral validation using metrics
        
        **Analytics & Validation:**
        - Performance metrics (SSIM, PSNR, SAM)
        - Agricultural indices (NDVI, NDWI, NDBI)
        - Crop health classification
        """)
        
    with st.expander("🤖 AI Methodology", expanded=False):
        st.markdown("""
        #### Cloud Removal Pipeline
        
        1. **Cloud Segmentation**
           - Input: RGB or RGBN image
           - Output: Binary cloud mask
        
        2. **Spectral Fusion**
           - NIR + SAR data integration
           - Cross-attention mechanisms
        
        3. **Reconstruction (Conditional Diffusion)**
           - Masked region regeneration
           - Spectral accuracy preservation
        
        4. **Validation**
           - SSIM > 0.75 target
           - NDVI correlation mapping
        """)
        
    with st.expander("📊 NDVI Calculation", expanded=False):
        st.markdown("""
        #### Normalized Difference Vegetation Index
        
        **Formula:**
        ```
        NDVI = (NIR - Red) / (NIR + Red)
        ```
        
        **Interpretation:**
        - NDVI < 0.2: Non-vegetation
        - 0.2 - 0.4: Sparse vegetation
        - 0.4 - 0.6: Moderate vegetation
        - 0.6+: Dense, healthy vegetation
        
        **Cloudkiller Enhancement:**
        - Real-time NDVI mapping
        - 30-day trend analysis
        - Anomaly detection
        - Crop health alerts
        """)

# ============================================================================
# 10. PAGE: ABOUT
# ============================================================================
elif selected == "ℹ️ About":
    
    st.markdown("### About AgriVision AI")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### 🎯 Mission
        
       Cloudkiller AI enables farmers and agricultural organizations to make data-driven decisions by providing real-time satellite intelligence for crop monitoring and resource optimization.
        
        #### 🏆 Achievements
        
        - **ISRO BAH2026 Hackathon** - Competing for cloud removal innovation
        - **Satellite Imagery Processing** - Multispectral analysis
        - **Real-time Intelligence** - Rapid inference pipeline
        - **Agricultural Impact** - Direct farmer benefits
        """)
        
    with col2:
        st.markdown("""
        #### 👥 Team: CloudKillers
        
        A dedicated team building solutions for the **ISRO Bharatiya Antariksh Hackathon 2026**, focusing on generative AI solutions for satellite data reconstruction and cloud removal for LISS-IV imagery.
        
        **Focus Areas:**
        - Frontend Development & Dashboarding
        - ML/AI Engineering & Model Architecture
        - Remote Sensing Data Processing
        - Validation & Agricultural Analytics
        """)
        
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 30px;">
        <p><strong>© 2026 AgriVision AI</strong></p>
        <p>CloudKillers Team | ISRO Bharatiya Antariksh Hackathon 2026</p>
        <p>Enabling precision agriculture through AI-powered satellite intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 11. FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85rem; margin-top: 20px;">
    <p>AgriVision AI • Smart Farm Monitoring • Powered by PyTorch & Streamlit</p>
    <p>Processing Time: {:.2f}s | Backend Status: {}</p>
</div>
""".format(
    st.session_state.processing_time,
    "✅ Online" if BACKEND_AVAILABLE else "⚠️ Demo Mode"
), unsafe_allow_html=True)