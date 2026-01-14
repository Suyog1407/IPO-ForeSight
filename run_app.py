#!/usr/bin/env python3
"""
Startup script for DRHP Analysis System
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

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

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn',
        'PyPDF2', 'pdfplumber', 'requests', 'beautifulsoup4',
        'transformers', 'sentence-transformers', 'nltk',
        'plotly', 'reportlab', 'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'data', 'data/uploads', 'data/processed', 'data/external',
        'models', 'reports', 'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 DRHP Analysis System - Starting Up")
    print("=" * 60)
    print(f"Startup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    print("Creating directories...")
    create_directories()
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Dependency check failed!")
        print("Please install missing packages and try again.")
        return False
    print()
    
    # Test system (moved to tests/)
    print("Testing system components...")
    try:
        test_script = os.path.join('tests', 'test_system.py')
        if os.path.exists(test_script):
            result = subprocess.run([sys.executable, test_script],
                                   capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("✅ System test passed!")
            else:
                print("⚠️ System test had issues, but continuing...")
                print(result.stdout)
        else:
            print("ℹ️ No tests found in tests/test_system.py, skipping")
    except Exception as e:
        print(f"⚠️ System test failed: {e}")
        print("Continuing anyway...")
    print()
    
    # Start Streamlit app (new path)
    print("Starting Streamlit application...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application.")
    print("=" * 60)
    
    try:
        # Run Streamlit app
        subprocess.run(['streamlit', 'run', 'src/ui/app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
