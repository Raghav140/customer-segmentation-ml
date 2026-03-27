"""
Launch script for the enhanced Streamlit app.
"""

import subprocess
import sys
from pathlib import Path

def launch_enhanced_app():
    """Launch the enhanced Streamlit app."""
    app_path = Path(__file__).parent / "app" / "enhanced_app.py"
    
    if not app_path.exists():
        print(f"❌ Enhanced app not found: {app_path}")
        return False
    
    print(f"🚀 Launching Enhanced Customer Segmentation App...")
    print(f"📱 App path: {app_path}")
    print(f"🌐 URL: http://localhost:8501")
    print("\n✨ Features available:")
    print("   ✅ Auto cluster selection")
    print("   ✅ API integration")
    print("   ✅ Model persistence")
    print("   ✅ Business constraints")
    print("   ✅ Enhanced UI/UX")
    print("\n🎯 Starting Streamlit...")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        return True
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to launch app: {str(e)}")
        return False

if __name__ == "__main__":
    success = launch_enhanced_app()
    sys.exit(0 if success else 1)
