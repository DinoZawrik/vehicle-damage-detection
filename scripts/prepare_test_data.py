"""
Prepare test data directory structure and README.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_SAMPLES_DIR = DATA_DIR / "test_samples"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DEMO_RESULTS_DIR = EXAMPLES_DIR / "demo_results"

def create_directories():
    """Create necessary directories."""
    dirs = [TEST_SAMPLES_DIR, DEMO_RESULTS_DIR]
    
    for d in dirs:
        if not d.exists():
            print(f"Creating directory: {d}")
            d.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Directory exists: {d}")

def create_readme():
    """Create README for test samples."""
    readme_path = TEST_SAMPLES_DIR / "README.md"
    
    content = """# Test Samples

Place your test images in this directory.
Supported formats: .jpg, .jpeg, .png

## Recommended Structure
- `car_01.jpg`
- `car_02.jpg`
- `annotations.json` (optional ground truth)

## Where to find images?
You can find free car damage images on:
- [Unsplash](https://unsplash.com/s/photos/car-damage)
- [Pexels](https://www.pexels.com/search/car%20accident/)
- [Kaggle Datasets](https://www.kaggle.com/datasets?search=car+damage)
"""
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created README: {readme_path}")

if __name__ == "__main__":
    print("📁 Setting up data directories...")
    create_directories()
    create_readme()
    print("✅ Setup complete!")
