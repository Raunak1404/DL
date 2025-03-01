import cv2
import numpy as np
import hashlib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Load ResNet50 model (without classification head)
model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

def generate_sha256(file_path):
    """
    Compute SHA-256 hash for exact match checking.
    """
    try:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"❌ Error generating SHA-256 hash: {e}")
        return None

def extract_features(img_path):
    """
    Extract deep features using ResNet50.
    Returns a 1D numpy feature vector.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features[0]
    except Exception as e:
        print(f"❌ Error extracting deep features: {e}")
        return None

def extract_video_features(video_path, frame_interval=10, max_frames=100):
    """
    Extract features from a video by sampling frames at set intervals.
    Returns a mean feature vector representing the video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    feature_list = []

    if not cap.isOpened():
        print("❌ Error: Unable to open video file.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames in the video
    if total_frames == 0:
        print("❌ Error: Video has no frames.")
        cap.release()
        return None

    while True:
        ret, frame = cap.read()
        if not ret:  # If no more frames, break the loop
            break

        if frame_count % frame_interval == 0:
            try:
                frame = cv2.resize(frame, (224, 224))
                frame = np.expand_dims(frame, axis=0).astype('float32')
                frame = preprocess_input(frame)
                feature = model.predict(frame)
                feature_list.append(feature[0])

                # Stop processing if we reach max_frames
                if len(feature_list) >= max_frames:
                    break
            except Exception as e:
                print(f"⚠️ Warning: Error processing frame {frame_count} - {e}")

        frame_count += 1

    cap.release()

    if feature_list:
        return np.mean(feature_list, axis=0)  # Average feature vector
    return None

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    print(f"🔍 Cosine Similarity: {similarity}")  # Log similarity score
    return similarity
