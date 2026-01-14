import os
import json
import logging
from typing import Dict, Any
from datetime import datetime

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        'data',
        'data/uploads',
        'data/processed',
        'data/external',
        'models',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json or create default"""
    config_path = 'config.json'
    
    default_config = {
        "app_name": "DRHP Analysis System",
        "version": "1.0.0",
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "supported_formats": ["pdf"],
        "analysis_settings": {
            "similarity_threshold": 0.8,
            "anomaly_threshold": 0.7,
            "min_section_length": 100
        },
        "external_sources": {
            "news_apis": [],
            "regulatory_sites": [],
            "company_databases": []
        },
        "chatbot_settings": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logging.warning(f"Error loading config: {e}. Using default config.")
            return default_config
    else:
        # Create default config file
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )

def save_analysis_results(results: Dict[str, Any], filename: str = None):
    """Save analysis results to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
    
    filepath = os.path.join('data/processed', filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return filepath

def load_analysis_results(filename: str) -> Dict[str, Any]:
    """Load analysis results from file"""
    filepath = os.path.join('data/processed', filename)
    
    with open(filepath, 'r') as f:
        return json.load(f)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def validate_pdf_file(file_path: str) -> bool:
    """Validate if file is a valid PDF"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            PyPDF2.PdfReader(file)
        return True
    except Exception:
        return False

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""
    stat = os.stat(file_path)
    
    return {
        'name': os.path.basename(file_path),
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'extension': os.path.splitext(file_path)[1].lower()
    }
