from flask import Flask, request, jsonify
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from utils import generate_sha256, extract_features, extract_video_features, cosine_similarity
from deepfake_detector import run

# 🔹 Firebase Initialization
SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase initialized.")

db = firestore.client()
app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify_media():
    """Verify the uploaded media for originality or deepfakes, returning a status label."""
    print("🚀 Received request for media verification.")
    file_path = None
    output_video_path = None
    try:
        # Ensure a file was provided in the request
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "Missing required parameters"}), 400

        file = request.files['file']
        media_type = request.form.get('media_type', 'unknown')

        # Save the uploaded file to a temporary location
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'tmp'
        file_path = f"temp_upload.{ext}"
        file.save(file_path)

        # Compute SHA-256 hash for the file
        file_hash_sha256 = generate_sha256(file_path)
        if not file_hash_sha256:
            return jsonify({"error": "Failed to generate SHA-256"}), 500

        # 🔹 **1️⃣ Check for Exact Match (by hash)**
        exact_match_docs = list(db.collection("media_hashes").where("hash", "==", file_hash_sha256).limit(1).stream())
        if exact_match_docs:
            original_hash = exact_match_docs[0].to_dict().get('hash')
            print(f"✅ Exact hash match found: {file_hash_sha256}")
            return jsonify({"status": "EDITED/COPIED", "original_hash": original_hash})

        # 🔹 **2️⃣ Extract Features for Similarity Check**
        if media_type == "image":
            new_features = extract_features(file_path)
        elif media_type == "video":
            new_features = extract_video_features(file_path)
        else:
            return jsonify({"error": "Unsupported media type"}), 400

        if new_features is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        # 🔹 **3️⃣ Check for Similar Content using Deep Feature Cosine Similarity**
        similar_docs = db.collection("media_hashes").stream()
        for doc in similar_docs:
            stored_data = doc.to_dict()
            stored_features = stored_data.get("feature_vector")
            if stored_features:
                stored_features = np.array([float(x) for x in stored_features])  # convert stored list to array
                similarity_score = cosine_similarity(new_features, stored_features)
                if similarity_score > 0.90:
                    print(f"⚠️ Similar content found! Similarity: {similarity_score}")
                    return jsonify({"status": "EDITED/COPIED", "original_hash": stored_data['hash']})

        # 🔹 **4️⃣ Run Deepfake Detection (for videos only)**
        if media_type == "video":
            output_video_path = "processed_video.mp4"
            deepfake_accuracy = run(file_path, output_video_path)
            print(f"🎭 Deepfake Detection Accuracy: {deepfake_accuracy}%")
            if deepfake_accuracy > 70:
                return jsonify({"status": "DEEPFAKE DETECTED", "accuracy": deepfake_accuracy})
            else:
                return jsonify({"status": "REAL VIDEO", "accuracy": deepfake_accuracy})

        # 🔹 **5️⃣ No issues found, mark as ORIGINAL**
        print("✅ No match found! Status: ORIGINAL")
        return jsonify({"status": "ORIGINAL"})
    finally:
        # 🔹 Cleanup: delete the temporary files after verification
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted temporary file: {file_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temp file {file_path}: {e}")
        if output_video_path and os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
                print(f"🗑️ Deleted temporary file: {output_video_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temp file {output_video_path}: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
