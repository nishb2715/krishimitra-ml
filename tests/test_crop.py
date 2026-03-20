import requests, os, sys

API = "http://localhost:8000"
TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_images')

def test_health():
    r = requests.get(f"{API}/health")
    assert r.status_code == 200
    print("Health check passed")

def test_crop_diagnose(image_path: str):
    with open(image_path, 'rb') as f:
        r = requests.post(
            f"{API}/crop/diagnose",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")}
        )
    assert r.status_code == 200
    data = r.json()
    print(f"\nCrop Test — {os.path.basename(image_path)}")
    print(f"  Predicted : {data['predicted_class']}")
    print(f"  Odia name : {data['odia_name']}")
    print(f"  Confidence: {data['confidence']}%")
    print(f"  Severity  : {data['severity']}")
    print(f"  Advice    : {data['advice_odia']}")
    print(f"  See vet   : {data['see_vet']}")
    print(f"  Top 3     : {data['top3']}")
    return data

if __name__ == "__main__":
    test_health()
    
    # Test with every image in tests/test_images/
    images = [f for f in os.listdir(TEST_DIR) 
              if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    if not images:
        print("Put some test images in tests/test_images/ and run again")
        sys.exit(1)
    
    for img in images:
        test_crop_diagnose(os.path.join(TEST_DIR, img))