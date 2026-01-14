import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Import our custom modules
from src.parsing.pdf_parser import PDFParser
from src.scraping.web_scraper import WebScraper
from src.anomaly.detector import AnomalyDetector
from src.plagiarism.detector import PlagiarismDetector
from src.chatbot import DRHPChatbot
from src.utils import create_directories, load_config

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

# Page configuration
st.set_page_config(
    page_title="IPO Foresight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    # Create necessary directories
    create_directories()
    
    # Load configuration
    config = load_config()
    
    # Sidebar navigation
    st.sidebar.title("📊 IPO Foresight")
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Dashboard", "📄 Document Upload", "🔍 Analysis", "🤖 ForeSight Bot", "📊 Reports"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📄 Document Upload":
        show_upload_page()
    elif page == "🔍 Analysis":
        show_analysis_page()
    elif page == "🤖 ForeSight Bot":
        show_chatbot_page()
    elif page == "📊 Reports":
        show_reports_page()

def show_dashboard():
    st.title("🏠 IPO Foresight Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", len(st.session_state.uploaded_files))
    
    with col2:
        st.metric("Anomalies Detected", 
                 len(st.session_state.analysis_results.get('anomalies', [])))
    
    with col3:
        st.metric("Plagiarism Cases", 
                 len(st.session_state.analysis_results.get('plagiarism', [])))
    
    with col4:
        st.metric("External Sources", 
                 len(st.session_state.analysis_results.get('external_data', [])))
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    if st.session_state.uploaded_files:
        df = pd.DataFrame(st.session_state.uploaded_files)
        st.dataframe(df, width='stretch')
    else:
        st.info("No documents uploaded yet. Go to Document Upload to get started!")

def show_upload_page():
    st.title("📄 Document Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload DRHP Document",
        type=['pdf'],
        help="Upload a PDF file containing the Draft Red Herring Prospectus"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = f"data/uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize PDF parser
        parser = PDFParser()
        
        with st.spinner("Processing document..."):
            # Extract sections
            sections = parser.extract_sections(file_path)
            
            # Store in session state
            file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'sections': list(sections.keys()),
                'path': file_path
            }
            
            st.session_state.uploaded_files.append(file_info)
            
            # Display results
            st.success(f"✅ Successfully processed {uploaded_file.name}")
            
            # Show extracted sections
            st.subheader("📋 Extracted Sections")
            for section, content in sections.items():
                with st.expander(f"📄 {section}"):
                    st.text(content[:500] + "..." if len(content) > 500 else content)
            
            # Store sections for analysis
            st.session_state.analysis_results['sections'] = sections

def show_analysis_page():
    st.title("🔍 Analysis & Detection")
    
    if not st.session_state.uploaded_files:
        st.warning("Please upload a document first!")
        return
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Run Anomaly Detection", type="primary"):
            with st.spinner("Detecting anomalies..."):
                detector = AnomalyDetector()
                anomalies = detector.detect_anomalies(st.session_state.analysis_results.get('sections', {}))
                st.session_state.analysis_results['anomalies'] = anomalies
                st.success(f"Found {len(anomalies)} anomalies")
    
    with col2:
        if st.button("🔍 Run Plagiarism Detection"):
            with st.spinner("Detecting plagiarism..."):
                plagiarism_detector = PlagiarismDetector()
                plagiarism_results = plagiarism_detector.detect_plagiarism(
                    st.session_state.analysis_results.get('sections', {})
                )
                st.session_state.analysis_results['plagiarism'] = plagiarism_results
                st.success(f"Found {len(plagiarism_results)} potential plagiarism cases")
    
    # New verification analysis
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🌐 Run External Verification"):
            with st.spinner("Cross-checking with external data..."):
                detector = AnomalyDetector()
                external_data = st.session_state.analysis_results.get('external_data', {})
                verification_anomalies = detector.detect_verification_anomalies(
                    st.session_state.analysis_results.get('sections', {}),
                    external_data
                )
                st.session_state.analysis_results['verification_anomalies'] = verification_anomalies
                st.success(f"Found {len(verification_anomalies)} verification discrepancies")
    
    with col4:
        if st.button("📊 Generate Location Map"):
            with st.spinner("Generating detailed location map..."):
                # Generate detailed location mapping for anomalies
                location_map = generate_anomaly_location_map(st.session_state.analysis_results.get('anomalies', []))
                st.session_state.analysis_results['location_map'] = location_map
                st.success("Location map generated")
    
    # Display results
    if st.session_state.analysis_results.get('anomalies'):
        st.subheader("🚨 Anomalies Detected")
        
        # Show location map if available
        if st.session_state.analysis_results.get('location_map'):
            location_map = st.session_state.analysis_results['location_map']
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("By Section", len(location_map['by_section']))
            with col2:
                st.metric("By Line", len(location_map['by_line']))
            with col3:
                st.metric("By Severity", len(location_map['by_severity']))
            
            # Detailed location table
            st.subheader("📍 Detailed Location Map")
            detailed_locations = location_map['detailed_locations']
            if detailed_locations:
                df = pd.DataFrame(detailed_locations)
                st.dataframe(df, width='stretch')
        
        # Individual anomalies with enhanced display
        for i, anomaly in enumerate(st.session_state.analysis_results['anomalies']):
            severity_icon = "🚨" if anomaly.get('severity') == 'high' else "⚠️" if anomaly.get('severity') == 'medium' else "ℹ️"
            
            with st.expander(f"{severity_icon} Anomaly {i+1}: {anomaly.get('type', 'Unknown')} - {anomaly.get('description', 'No description')[:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"Type: {anomaly.get('type', 'Unknown')}")
                    st.write(f"Severity: {anomaly.get('severity', 'Unknown')}")
                    st.write(f"Section: {anomaly.get('section', 'Unknown')}")
                    if anomaly.get('line_number'):
                        st.write(f"Line Number: {anomaly.get('line_number')}")
                
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
    
    # Display verification anomalies
    if st.session_state.analysis_results.get('verification_anomalies'):
        st.subheader("🌐 External Verification Results")
        for i, verification in enumerate(st.session_state.analysis_results['verification_anomalies']):
            with st.expander(f"Verification {i+1}: {verification.get('description', 'No description')[:50]}..."):
                st.write(f"**Discrepancy:** {verification.get('discrepancy_percentage', 0):.1%}")
                st.write(f"**DRHP Claim:** {verification.get('drhp_claim', {})}")
                st.write(f"**External Data:** {verification.get('news_data', verification.get('website_data', {}))}")
                st.write(f"**Recommendation:** {verification.get('recommendation', 'No recommendation')}")
    
    if st.session_state.analysis_results.get('plagiarism'):
        st.subheader("⚠️ Plagiarism Detected")
        for i, case in enumerate(st.session_state.analysis_results['plagiarism']):
            with st.expander(f"Plagiarism Case {i+1}: {case.get('similarity_score', 0):.2f}% similarity"):
                st.json(case)

def show_chatbot_page():
    st.title("🤖 ForeSight Bot")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DRHPChatbot()
    
    # Chat interface
    st.subheader("💬 Ask questions about your DRHP analysis")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about anomalies, plagiarism, or any DRHP insights..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get chatbot response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.get_response(
                prompt, 
                st.session_state.analysis_results
            )
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display response
        with st.chat_message("assistant"):
            st.write(response)

def show_reports_page():
    st.title("📊 Reports & Export")
    
    if not st.session_state.analysis_results:
        st.warning("No analysis results available. Please run analysis first!")
        return
    
    # Generate report options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate PDF Report"):
            st.info("PDF report generation feature coming soon!")
    
    with col2:
        if st.button("📊 Generate Excel Report"):
            st.info("Excel report generation feature coming soon!")
    
    # Display summary statistics
    st.subheader("📈 Analysis Summary")
    
    # Create visualizations
    if st.session_state.analysis_results.get('anomalies'):
        fig = px.bar(
            x=[a.get('type', 'Unknown') for a in st.session_state.analysis_results['anomalies']],
            title="Anomaly Types Distribution"
        )
        st.plotly_chart(fig, width='stretch')
    
    if st.session_state.analysis_results.get('plagiarism'):
        similarity_scores = [p.get('similarity_score', 0) for p in st.session_state.analysis_results['plagiarism']]
        fig = px.histogram(
            x=similarity_scores,
            title="Plagiarism Similarity Score Distribution",
            labels={'x': 'Similarity Score (%)', 'y': 'Count'}
        )
        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()
