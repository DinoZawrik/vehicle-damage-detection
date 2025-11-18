import os
import sys
import urllib.request
import subprocess
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "demo_images"
OUTPUT_DIR = ROOT_DIR / "assets" / "screenshots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image sources
IMAGES = {
    "demo_car_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg",
    "demo_crash_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Audi_V8_accident.jpg"
}

def download_image(url, path):
    print(f"Downloading {url} to {path}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def run_demo(image_name):
    input_path = DATA_DIR / image_name
    output_path = OUTPUT_DIR / f"result_{image_name}"
    
    print(f"Running demo on {input_path}...")
    
    cmd = [
        sys.executable,
        str(ROOT_DIR / "demo.py"),
        "--image", str(input_path),
        "--output", str(output_path),
        "--simulate",
        "--no-comparison"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Success! Saved to {output_path}")
    else:
        print("Error running demo:")
        print(result.stderr)

def main():
    print("Generating demo assets...")
    
    for name, url in IMAGES.items():
        path = DATA_DIR / name
        if not path.exists():
            if download_image(url, path):
                run_demo(name)
        else:
            run_demo(name)
            
    print("Done!")

if __name__ == "__main__":
    main()
