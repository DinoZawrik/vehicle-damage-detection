"""
Download YOLOv8 model for Vehicle Damage Detection MVP.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

def ensure_models_dir():
    """Ensure models directory exists."""
    if not MODELS_DIR.exists():
        print(f"Creating models directory: {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_model():
    """Download YOLOv8n model if not exists."""
    ensure_models_dir()
    
    model_path = MODELS_DIR / MODEL_FILENAME
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return True
    
    print(f"⬇️  Downloading YOLOv8n model from {MODEL_URL}...")
    print(f"   Target: {model_path}")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r   Progress: {percent}%")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(MODEL_URL, str(model_path), progress_hook)
        print("\n✅ Download complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Vehicle Damage Detection - Model Setup")
    print("=" * 40)
    
    if download_model():
        print("\n✨ Model setup successful!")
        sys.exit(0)
    else:
        print("\n💥 Model setup failed!")
        sys.exit(1)
