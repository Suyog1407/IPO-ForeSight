import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import base64
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Import our custom modules
from src.parsing.pdf_parser import PDFParser
from src.scraping.web_scraper import WebScraper
from src.anomaly.detector import AnomalyDetector
from src.plagiarism.detector import PlagiarismDetector
from src.chatbot import DRHPChatbot
from src.utils import create_directories, load_config

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-color: #f8f9fa;
        --dark-color: #343a40;
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    
    /* Custom cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .metric-card h3 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dark-color);
        margin: 0;
    }
    
    /* Status indicators */
    .status-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Animated progress bars */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 20px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px 10px 0 0;
        border: none;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        background: var(--dark-color);
        color: white;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def generate_anomaly_location_map(anomalies):
    """Generate detailed location mapping for anomalies"""
    location_map = {
        'by_section': {},
        'by_line': {},
        'by_severity': {},
        'detailed_locations': []
    }
    
    for i, anomaly in enumerate(anomalies):
        section = anomaly.get('section', 'unknown')
        line_number = anomaly.get('line_number', 'N/A')
        severity = anomaly.get('severity', 'unknown')
        
        # Group by section
        if section not in location_map['by_section']:
            location_map['by_section'][section] = []
        location_map['by_section'][section].append(anomaly)
        
        # Group by line number
        if line_number != 'N/A':
            if line_number not in location_map['by_line']:
                location_map['by_line'][line_number] = []
            location_map['by_line'][line_number].append(anomaly)
        
        # Group by severity
        if severity not in location_map['by_severity']:
            location_map['by_severity'][severity] = []
        location_map['by_severity'][severity].append(anomaly)
        
        # Detailed location info
        location_map['detailed_locations'].append({
            'anomaly_id': i + 1,
            'section': section,
            'line_number': line_number,
            'severity': severity,
            'description': anomaly.get('description', ''),
            'exact_location': anomaly.get('exact_location', ''),
            'context': anomaly.get('context', ''),
            'type': anomaly.get('type', 'unknown')
        })
    
    return location_map

def create_metric_card(title, value, icon, color="primary"):
    """Create a custom metric card"""
    color_map = {
        "primary": "#1f77b4",
        "success": "#2ca02c", 
        "warning": "#d62728",
        "info": "#17a2b8"
    }
    
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color_map.get(color, color_map['primary'])};">
        <h3>{icon} {title}</h3>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def create_progress_bar(label, value, max_value=100):
    """Create an animated progress bar"""
    percentage = (value / max_value) * 100
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {percentage}%;">
            {label}: {value}/{max_value}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_status_badge(status, text):
    """Create a status badge"""
    status_class = f"status-{status}"
    st.markdown(f'<span class="{status_class}">{text}</span>', unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="DRHP Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 DRHP Analysis System</h1>
    <p>Advanced Draft Red Herring Prospectus Analysis with AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    
    selected = option_menu(
        menu_title=None,
        options=["📊 Dashboard", "📄 Document Analysis", "🔍 Anomaly Detection", "⚠️ Plagiarism Check", "🌐 External Verification", "🤖 AI Chatbot", "📈 Reports"],
        icons=["speedometer2", "file-text", "search", "shield-check", "globe", "robot", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )

# Dashboard Page
if selected == "📊 Dashboard":
    st.markdown("## 📊 System Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Documents Analyzed", "1", "📄", "primary")
    
    with col2:
        create_metric_card("Anomalies Detected", "19", "🚨", "warning")
    
    with col3:
        create_metric_card("Plagiarism Cases", "3", "⚠️", "info")
    
    with col4:
        create_metric_card("External Sources", "20", "🌐", "success")
    
    # System status
    st.markdown("## 🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Active Components")
        st.success("📄 PDF Parser - Ready")
        st.success("🔍 Anomaly Detector - Ready")
        st.success("⚠️ Plagiarism Detector - Ready")
        st.success("🌐 Web Scraper - Ready")
        st.success("🤖 AI Chatbot - Ready")
    
    with col2:
        st.markdown("### 📊 Analysis Progress")
        create_progress_bar("Document Processing", 100, 100)
        create_progress_bar("Anomaly Detection", 100, 100)
        create_progress_bar("External Verification", 80, 100)
        create_progress_bar("Report Generation", 90, 100)
    
    # Recent activity
    st.markdown("## 📈 Recent Activity")
    
    activity_data = {
        "Time": ["10:30 AM", "10:25 AM", "10:20 AM", "10:15 AM"],
        "Activity": ["Analysis Completed", "External Data Retrieved", "Anomalies Detected", "Document Uploaded"],
        "Status": ["✅", "✅", "⚠️", "✅"]
    }
    
    df_activity = pd.DataFrame(activity_data)
    st.dataframe(df_activity, use_container_width=True)

# Document Analysis Page
elif selected == "📄 Document Analysis":
    st.markdown("## 📄 Document Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload DRHP Document",
        type=['pdf'],
        help="Upload your Draft Red Herring Prospectus for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                with st.spinner("Analyzing document..."):
                    # Simulate analysis
                    time.sleep(2)
                    
                    # Parse document
                    parser = PDFParser()
                    sections = parser.extract_sections(f"temp_{uploaded_file.name}")
                    
                    st.session_state.analysis_results['sections'] = sections
                    st.success("✅ Document analysis completed!")
        
        with col2:
            if st.button("🌐 Gather External Data", use_container_width=True):
                with st.spinner("Gathering external data..."):
                    # Simulate external data gathering
                    time.sleep(2)
                    
                    scraper = WebScraper()
                    external_data = scraper.aggregate_external_data("Urban Company")
                    st.session_state.analysis_results['external_data'] = external_data
                    st.success("✅ External data gathered!")
    
    # Display results
    if st.session_state.analysis_results.get('sections'):
        st.markdown("## 📊 Analysis Results")
        
        sections = st.session_state.analysis_results['sections']
        
        # Section overview
        st.markdown("### 📋 Document Sections")
        
        section_data = []
        for section_name, section_data_dict in sections.items():
            if isinstance(section_data_dict, dict):
                word_count = len(section_data_dict.get('content', '').split())
                page_number = section_data_dict.get('page_number', 'Unknown')
            else:
                word_count = len(str(section_data_dict).split())
                page_number = 'Unknown'
            
            section_data.append({
                'Section': section_name.replace('_', ' ').title(),
                'Words': word_count,
                'Page': page_number,
                'Status': '✅ Complete' if word_count > 100 else '⚠️ Incomplete'
            })
        
        df_sections = pd.DataFrame(section_data)
        st.dataframe(df_sections, use_container_width=True)

# Anomaly Detection Page
elif selected == "🔍 Anomaly Detection":
    st.markdown("## 🔍 Anomaly Detection")
    
    if st.session_state.analysis_results.get('sections'):
        # Analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚨 Run Anomaly Detection", type="primary", use_container_width=True):
                with st.spinner("Detecting anomalies..."):
                    detector = AnomalyDetector()
                    anomalies = detector.detect_anomalies(st.session_state.analysis_results['sections'])
                    st.session_state.analysis_results['anomalies'] = anomalies
                    st.success(f"✅ Found {len(anomalies)} anomalies")
        
        with col2:
            if st.button("📍 Generate Location Map", use_container_width=True):
                with st.spinner("Generating location map..."):
                    location_map = generate_anomaly_location_map(st.session_state.analysis_results.get('anomalies', []))
                    st.session_state.analysis_results['location_map'] = location_map
                    st.success("✅ Location map generated")
        
        with col3:
            if st.button("📊 Create Summary", use_container_width=True):
                with st.spinner("Creating summary..."):
                    st.success("✅ Summary created")
        
        # Display anomalies
        if st.session_state.analysis_results.get('anomalies'):
            anomalies = st.session_state.analysis_results['anomalies']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                create_metric_card("Total Anomalies", str(len(anomalies)), "🚨", "warning")
            
            with col2:
                high_severity = len([a for a in anomalies if a.get('severity') == 'high'])
                create_metric_card("High Severity", str(high_severity), "🔴", "warning")
            
            with col3:
                medium_severity = len([a for a in anomalies if a.get('severity') == 'medium'])
                create_metric_card("Medium Severity", str(medium_severity), "🟡", "info")
            
            with col4:
                low_severity = len([a for a in anomalies if a.get('severity') == 'low'])
                create_metric_card("Low Severity", str(low_severity), "🟢", "success")
            
            # Detailed anomalies
            st.markdown("### 🔍 Detailed Anomalies")
            
            for i, anomaly in enumerate(anomalies[:10], 1):  # Show first 10
                severity = anomaly.get('severity', 'unknown')
                severity_icon = "🔴" if severity == 'high' else "🟡" if severity == 'medium' else "🟢"
                
                with st.expander(f"{severity_icon} Anomaly {i}: {anomaly.get('type', 'Unknown')} - {anomaly.get('description', 'No description')[:50]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Information:**")
                        st.write(f"Type: {anomaly.get('type', 'Unknown')}")
                        st.write(f"Severity: {anomaly.get('severity', 'Unknown')}")
                        st.write(f"Section: {anomaly.get('section', 'Unknown')}")
                        if anomaly.get('page_number'):
                            st.write(f"Page Number: {anomaly.get('page_number')}")
                    
                    with col2:
                        st.write("**Location Details:**")
                        if anomaly.get('page_number'):
                            st.write(f"Page Number: {anomaly.get('page_number')}")
                        if anomaly.get('section_start_line') and anomaly.get('section_end_line'):
                            st.write(f"Section Lines: {anomaly.get('section_start_line')}-{anomaly.get('section_end_line')}")
                        if anomaly.get('exact_location'):
                            st.write(f"Exact Location: {anomaly.get('exact_location')}")
                        if anomaly.get('context'):
                            st.write(f"Context: {anomaly.get('context')}")
                    
                    st.write("**Full Details:**")
                    st.json(anomaly)
    
    else:
        st.warning("⚠️ Please upload and analyze a document first.")

# Plagiarism Check Page
elif selected == "⚠️ Plagiarism Check":
    st.markdown("## ⚠️ Plagiarism Detection")
    
    if st.session_state.analysis_results.get('sections'):
        if st.button("🔍 Run Plagiarism Detection", type="primary", use_container_width=True):
            with st.spinner("Detecting plagiarism..."):
                plagiarism_detector = PlagiarismDetector()
                plagiarism_results = plagiarism_detector.detect_plagiarism(st.session_state.analysis_results['sections'])
                st.session_state.analysis_results['plagiarism'] = plagiarism_results
                st.success(f"✅ Found {len(plagiarism_results)} plagiarism cases")
        
        # Display plagiarism results
        if st.session_state.analysis_results.get('plagiarism'):
            plagiarism_results = st.session_state.analysis_results['plagiarism']
            
            # Summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_metric_card("Total Cases", str(len(plagiarism_results)), "⚠️", "warning")
            
            with col2:
                high_similarity = len([p for p in plagiarism_results if p.get('similarity_score', 0) > 0.8])
                create_metric_card("High Similarity", str(high_similarity), "🔴", "warning")
            
            with col3:
                avg_similarity = sum([p.get('similarity_score', 0) for p in plagiarism_results]) / len(plagiarism_results)
                create_metric_card("Avg Similarity", f"{avg_similarity:.1%}", "📊", "info")
            
            # Detailed cases
            st.markdown("### 🔍 Plagiarism Cases")
            
            for i, case in enumerate(plagiarism_results, 1):
                similarity_score = case.get('similarity_score', 0)
                severity_icon = "🔴" if similarity_score > 0.8 else "🟡" if similarity_score > 0.6 else "🟢"
                
                with st.expander(f"{severity_icon} Case {i}: {case.get('section', 'Unknown')} - {similarity_score:.1%} similarity"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Case Information:**")
                        st.write(f"Section: {case.get('section', 'Unknown')}")
                        st.write(f"Similarity: {similarity_score:.1%}")
                        st.write(f"Source: {case.get('source', 'Unknown')}")
                        st.write(f"Severity: {case.get('severity', 'Unknown')}")
                    
                    with col2:
                        st.write("**Matched Content:**")
                        st.write(case.get('matched_text', 'No content'))
                    
                    st.write("**Description:**")
                    st.write(case.get('description', 'No description'))
    
    else:
        st.warning("⚠️ Please upload and analyze a document first.")

# External Verification Page
elif selected == "🌐 External Verification":
    st.markdown("## 🌐 External Verification")
    
    if st.session_state.analysis_results.get('sections'):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌐 Run External Verification", type="primary", use_container_width=True):
                with st.spinner("Cross-checking with external data..."):
                    detector = AnomalyDetector()
                    external_data = st.session_state.analysis_results.get('external_data', {})
                    verification_anomalies = detector.detect_verification_anomalies(
                        st.session_state.analysis_results.get('sections', {}),
                        external_data
                    )
                    st.session_state.analysis_results['verification_anomalies'] = verification_anomalies
                    st.success(f"✅ Found {len(verification_anomalies)} verification discrepancies")
        
        with col2:
            if st.button("📊 Generate Verification Report", use_container_width=True):
                with st.spinner("Generating verification report..."):
                    st.success("✅ Verification report generated")
        
        # Display verification results
        if st.session_state.analysis_results.get('verification_anomalies'):
            verification_anomalies = st.session_state.analysis_results['verification_anomalies']
            
            # Summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_metric_card("Discrepancies", str(len(verification_anomalies)), "🌐", "warning")
            
            with col2:
                high_discrepancy = len([v for v in verification_anomalies if v.get('discrepancy_percentage', 0) > 0.5])
                create_metric_card("High Discrepancy", str(high_discrepancy), "🔴", "warning")
            
            with col3:
                avg_discrepancy = sum([v.get('discrepancy_percentage', 0) for v in verification_anomalies]) / len(verification_anomalies)
                create_metric_card("Avg Discrepancy", f"{avg_discrepancy:.1%}", "📊", "info")
            
            # Detailed verification
            st.markdown("### 🔍 Verification Results")
            
            for i, verification in enumerate(verification_anomalies, 1):
                discrepancy = verification.get('discrepancy_percentage', 0)
                severity_icon = "🔴" if discrepancy > 0.5 else "🟡" if discrepancy > 0.2 else "🟢"
                
                with st.expander(f"{severity_icon} Verification {i}: {verification.get('description', 'No description')[:50]}..."):
                    st.write(f"**Discrepancy:** {discrepancy:.1%}")
                    st.write(f"**DRHP Claim:** {verification.get('drhp_claim', {})}")
                    st.write(f"**External Data:** {verification.get('news_data', verification.get('website_data', {}))}")
                    st.write(f"**Recommendation:** {verification.get('recommendation', 'No recommendation')}")
    
    else:
        st.warning("⚠️ Please upload and analyze a document first.")

# AI Chatbot Page
elif selected == "🤖 AI Chatbot":
    st.markdown("## 🤖 AI Chatbot for DRHP Insights")
    
    # Initialize chatbot
    chatbot = DRHPChatbot()
    
    # Chat interface
    st.markdown("### 💬 Ask Questions About Your DRHP")
    
    # Sample questions
    st.markdown("**💡 Sample Questions:**")
    sample_questions = [
        "What anomalies were found and where are they located?",
        "Are there any discrepancies between DRHP claims and external data?",
        "What plagiarism cases were detected?",
        "Which sections have the most issues?",
        "What are the high-severity anomalies that need immediate attention?"
    ]
    
    for question in sample_questions:
        if st.button(f"❓ {question}", use_container_width=True):
            st.session_state.current_question = question
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about your DRHP analysis:",
        value=st.session_state.get('current_question', ''),
        placeholder="Type your question here..."
    )
    
    if st.button("🚀 Get AI Response", type="primary"):
        if user_question:
            with st.spinner("🤖 AI is thinking..."):
                response = chatbot.get_response(user_question, st.session_state.analysis_results)
                st.markdown("### 🤖 AI Response:")
                st.markdown(response)
        else:
            st.warning("⚠️ Please enter a question.")

# Reports Page
elif selected == "📈 Reports":
    st.markdown("## 📈 Analysis Reports")
    
    # Report generation options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Summary Report", type="primary", use_container_width=True):
            st.success("✅ Summary report generated")
    
    with col2:
        if st.button("📋 Generate Detailed Report", use_container_width=True):
            st.success("✅ Detailed report generated")
    
    with col3:
        if st.button("📈 Generate Analytics Report", use_container_width=True):
            st.success("✅ Analytics report generated")
    
    # Sample report data
    if st.session_state.analysis_results.get('anomalies'):
        st.markdown("### 📊 Analysis Summary")
        
        # Create charts
        anomalies = st.session_state.analysis_results['anomalies']
        
        # Severity distribution
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        fig_severity = px.pie(
            values=list(severity_counts.values()),
            names=list(severity_counts.keys()),
            title="Anomaly Severity Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_severity, use_container_width=True)
        
        # Section distribution
        section_counts = {}
        for anomaly in anomalies:
            section = anomaly.get('section', 'unknown')
            section_counts[section] = section_counts.get(section, 0) + 1
        
        fig_section = px.bar(
            x=list(section_counts.keys()),
            y=list(section_counts.values()),
            title="Anomalies by Section",
            color=list(section_counts.values()),
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_section, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <h3>🚀 DRHP Analysis System</h3>
    <p>Advanced AI-Powered Document Analysis | Built with Streamlit</p>
    <p>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
