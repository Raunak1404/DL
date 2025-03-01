import cv2
import torch
import logging
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os

def detect_deepfake(video_path: str, output_path: str = None, sample_interval: int = 5, embedding_threshold: float = 0.6) -> str:
    """
    Analyze a video to detect deepfakes based on face embedding consistency.
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path for saving the processed output video with highlights.
                            If None, a default "<video_name>_output<ext>" will be used.
        sample_interval (int): Process every N-th frame for efficiency (default=5).
        embedding_threshold (float): Cosine similarity threshold to flag an inconsistent face (default=0.6).
    Returns:
        str: "Deepfake Detected" if a deepfake is likely, otherwise "Real Video".
    """
    # Initialize face detection (MTCNN) and face embedding (InceptionResnetV1) models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")
    
    # Determine output video path
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        if ext == '':
            ext = '.mp4'  # default extension if none
        output_path = base + '_output' + ext
    
    # Prepare video writer for output video
    fps = cap.get(cv2.CAP_PROP_FPS) or 0  # FPS may be 0 if unavailable
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Choose codec based on output file extension
    ext_lower = os.path.splitext(output_path)[1].lower()
    if ext_lower in ['.avi', '.divx']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Unable to open video writer for output file: {output_path}")
    
    logging.info(f"Processing video: {video_path}")
    logging.info(f"Frame sampling interval: {sample_interval}")
    
    # Variables to track analysis
    baseline_embedding = None           # embedding of the first detected face (baseline)
    processed_faces_count = 0           # number of frames where a face was detected and processed
    flagged_count = 0                   # number of frames flagged as deepfake (inconsistent face)
    frame_index = 0
    
    try:
        # Process frames with no gradient calculation (inference mode)
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                if frame_index % sample_interval == 0:
                    # Convert frame (OpenCV BGR) to PIL Image (RGB) for face detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    # Detect faces in this frame
                    boxes, probs = mtcnn.detect(pil_img)
                    if boxes is not None and len(boxes) > 0:
                        # If multiple faces are detected, select the largest face
                        if len(boxes) > 1:
                            areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
                            idx = int(np.argmax(areas))
                        else:
                            idx = 0
                        x1, y1, x2, y2 = boxes[idx]
                        # Extract the face region and prepare it for embedding
                        face_pil = pil_img.crop((x1, y1, x2, y2))
                        face_pil = face_pil.resize((160, 160))
                        face_tensor = F.to_tensor(face_pil)
                        face_tensor = F.normalize(face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        face_tensor = face_tensor.unsqueeze(0).to(device)
                        # Compute the face embedding vector
                        embedding = resnet(face_tensor)
                        face_tensor = None  # free tensor memory
                        
                        processed_faces_count += 1
                        if baseline_embedding is None:
                            # Set baseline embedding from the first face
                            baseline_embedding = embedding.clone()
                            logging.debug(f"Frame {frame_index}: Baseline face embedding established.")
                        else:
                            # Compare current embedding to baseline using cosine similarity
                            sim = torch.nn.functional.cosine_similarity(embedding, baseline_embedding).item()
                            if sim < embedding_threshold:
                                flagged_count += 1
                                # Highlight this frame: draw red rectangle and label
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(frame, "Deepfake", (int(x1), max(0, int(y1) - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                logging.debug(f"Frame {frame_index}: similarity={sim:.2f} (FLAGGED as deepfake).")
                            else:
                                logging.debug(f"Frame {frame_index}: similarity={sim:.2f} (consistent).")
                        embedding = None  # free embedding tensor (baseline is kept)
                    else:
                        logging.debug(f"Frame {frame_index}: No face detected.")
                # Write the frame (with any highlights) to the output video
                writer.write(frame)
                frame_index += 1
    except Exception as e:
        logging.exception(f"Error during processing: {e}")
        raise
    finally:
        # Release video resources
        cap.release()
        writer.release()
        # Clean up models and GPU memory
        if 'mtcnn' in locals():
            del mtcnn
        if 'resnet' in locals():
            del resnet
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Determine final classification based on flagged frames ratio
    result = "Real Video"
    if processed_faces_count == 0:
        logging.warning("No faces detected in the video; unable to perform deepfake analysis.")
        # No face data – classify as real (or inconclusive) by default
    elif processed_faces_count > 1:
        # Exclude the baseline frame from ratio (only compare subsequent frames)
        inconsistency_ratio = flagged_count / (processed_faces_count - 1)
        logging.info(f"Analyzed frames (excluding baseline): {processed_faces_count - 1}, "
                     f"Frames flagged: {flagged_count} ({inconsistency_ratio*100:.1f}% inconsistent)")
        if inconsistency_ratio > 0.5:
            result = "Deepfake Detected"
    else:
        # Only one face frame found (no comparisons possible)
        logging.info("Only one face frame detected; no frame-to-frame comparison available.")
    logging.info(f"Deepfake detection result: {result}")
    return result

# Allow the module to be run as a script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect deepfake in a video by analyzing face consistency.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("-o", "--output", help="Path to save the output video with highlights (default: '<video>_output.ext')")
    parser.add_argument("-s", "--skip", type=int, default=5, help="Frame sampling interval (default: 5 frames)")
    parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Cosine similarity threshold for flagging inconsistencies (default: 0.6)")
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    try:
        result = detect_deepfake(args.video, args.output, sample_interval=args.skip, embedding_threshold=args.threshold)
        print(result)
    except Exception as e:
        logging.error(f"Deepfake detection failed: {e}")
        exit(1)
run = detect_deepfake