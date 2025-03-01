import firebase_admin
from firebase_admin import credentials, firestore, auth
import os

# ✅ Check if serviceAccountKey.json exists before proceeding
SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"
if not os.path.exists(SERVICE_ACCOUNT_FILE):
    print(f"❌ Missing {SERVICE_ACCOUNT_FILE}. Please add it to the project folder.")
    exit(1)

# ✅ Initialize Firebase if not already initialized
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized successfully.")
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        exit(1)

# ✅ Connect to Firestore
try:
    db = firestore.client()
    print("📡 Firestore connected successfully.")
except Exception as e:
    print(f"❌ Firestore connection failed: {e}")
    exit(1)

# ✅ Test Firestore connection
def test_firestore():
    try:
        print("📝 Writing test document to Firestore...")
        doc_ref = db.collection("test").document("connection_test")
        doc_ref.set({"message": "Firestore connection successful!"})
        print("✅ Firestore Connected Successfully! Data written.")
    except Exception as e:
        print(f"❌ Firestore Connection Failed: {e}")

# ✅ Run Firestore Test
if __name__ == "__main__":
    test_firestore()
