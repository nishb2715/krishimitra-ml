import requests, os, sys

API = "http://localhost:8000"
TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_images')

def test_livestock_diagnose(image_path: str):
    with open(image_path, 'rb') as f:
        r = requests.post(
            f"{API}/livestock/diagnose",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")}
        )
    assert r.status_code == 200
    data = r.json()
    print(f"\nLivestock Test — {os.path.basename(image_path)}")
    print(f"  Predicted : {data['predicted_class']}")
    print(f"  Odia name : {data['odia_name']}")
    print(f"  Confidence: {data['confidence']}%")
    print(f"  Advice    : {data['advice_odia']}")
    print(f"  Urgent vet: {data['see_vet_urgently']}")
    return data

if __name__ == "__main__":
    images = [f for f in os.listdir(TEST_DIR)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    if not images:
        print("Put some test images in tests/test_images/ and run again")
        sys.exit(1)
    
    for img in images:
        test_livestock_diagnose(os.path.join(TEST_DIR, img))