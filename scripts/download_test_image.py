"""
Download a real test image with proper headers.
"""
import urllib.request
import sys

def download_image():
    url = "https://ultralytics.com/images/bus.jpg"
    output = "data/test_samples/real_damage.jpg"
    
    print(f"Downloading {url}...")
    
    req = urllib.request.Request(
        url, 
        data=None, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response, open(output, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            print(f"✅ Saved to {output}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_image()
