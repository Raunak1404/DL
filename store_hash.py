import firebase_admin
import requests
import datetime
import os
from firebase_setup import db  # Firestore setup
from utils import generate_sha256, extract_features, extract_video_features

app = None  # Flask not required for CLI input
VERIFY_API_URL = "http://127.0.0.1:5000/verify"  # Endpoint for media verification

# Ensure Firebase app is initialized
if not firebase_admin._apps:
    print("❌ Firebase not initialized. Run `python firebase_setup.py` first!")
    exit(1)

def store_hash(file_path, media_type):
    """
    Receives a media file path from the user, processes it,
    verifies authenticity via the `verify_api.py`, and stores metadata in Firestore.
    The actual media file is deleted after processing.
    """

    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    print(f"📂 Processing file: {file_path}")

    try:
        # Compute SHA-256 hash
        file_hash_sha256 = generate_sha256(file_path)
        if not file_hash_sha256:
            print("❌ Error generating SHA-256 hash.")
            return

        # Extract deep features
        if media_type == "image":
            deep_features = extract_features(file_path)
        elif media_type == "video":
            deep_features = extract_video_features(file_path)
        else:
            print(f"❌ Unsupported media type: {media_type}")
            return

        if deep_features is None:
            print(f"❌ Error extracting features for {media_type}.")
            return

        # 🔹 Send file to `verify_api.py`
        with open(file_path, "rb") as f:
            response = requests.post(
                VERIFY_API_URL,
                files={"file": f},
                data={"media_type": media_type}
            )

        if response.status_code != 200:
            print("❌ Error: Failed to verify media.")
            return

        result = response.json()
        status_tag = result.get("status", "ORIGINAL")
        original_reference_id = result.get("original_hash", None)

        # 🔹 Store metadata in Firestore (no actual media stored)
        store_data = {
            "hash": file_hash_sha256,
            "media_type": media_type,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "status": status_tag,
            "original_reference_id": original_reference_id,
            "feature_vector": deep_features.tolist()
        }

        try:
            db.collection("media_hashes").add(store_data)
            print(f"✅ {media_type.capitalize()} stored successfully with status '{status_tag}'!")
        except Exception as e:
            print(f"❌ Error storing {media_type} metadata in Firestore: {e}")
    finally:
        # 🔹 Cleanup: Delete the temporary file after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted temporary file: {file_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temporary file {file_path}: {e}")

if __name__ == "__main__":
    # Prompt user for file path manually
    file_path = input("Enter the path of the image/video: ").strip()

    if not os.path.exists(file_path):
        print("❌ File not found! Please check the path and try again.")
        exit(1)

    # Determine media type from file extension
    ext = file_path.lower().split('.')[-1]
    media_type = "video" if ext in ["mp4", "mov", "avi", "mkv"] else "image"

    # Process the media file
    store_hash(file_path, media_type)
