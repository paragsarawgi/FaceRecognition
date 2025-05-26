import cv2
import face_recognition
import os
import numpy as np

# Load known face encodings and names
from PIL import Image

def load_known_faces(directory="A:\\Visual Studio Code\\Code Playground\\facerecog\\known_faces"):
    known_encodings = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory, filename)
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    image_np = np.array(img)
                    image_np = np.ascontiguousarray(image_np)
                    # Resize large images to something manageable, e.g. width=800
                    h, w = image_np.shape[:2]
                    if w > 800:
                        scale = 800 / w
                        image_np = cv2.resize(image_np, (800, int(h * scale)))
                    encodings = face_recognition.face_encodings(image_np)
                    print(f"[DEBUG] {filename}: dtype={image_np.dtype}, shape={image_np.shape}")

                    # Check again just to be safe
                    if image_np.dtype != np.uint8 or image_np.shape[2] != 3:
                        print(f"[Error] {filename}: Invalid format for face_recognition")
                        continue

                    encodings = face_recognition.face_encodings(image_np)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(os.path.splitext(filename)[0])
                    else:
                        print(f"[Warning] No face found in {filename}")
            except Exception as e:
                print(f"[Error] Failed to process {filename}: {e}")
    return known_encodings, known_names



# Load known faces
known_encodings, known_names = load_known_faces()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Dictionary to store face detection counts
detection_counts = {}
THRESHOLD = 5  # Number of times a face must be detected to be displayed

while cap.isOpened():
    ret, frame = cap.read()  # FIXED: should be cap.read() not video_capture.read()

    if not ret or frame is None:
        print("[Error] Failed to capture frame from webcam.")
        continue  # ✅ now correctly inside the while loop

    # Ensure the frame is uint8 RGB
    frame = frame.astype(np.uint8)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)


    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    detected_faces = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        
        detected_faces.append(name)
        
        # Count detections for each face
        detection_counts[name] = detection_counts.get(name, 0) + 1
        
        # Only display the name if it has been detected at least THRESHOLD times
        if detection_counts[name] >= THRESHOLD:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Reset detection counts for undetected faces
    for name in list(detection_counts.keys()):
        if name not in detected_faces:
            detection_counts[name] = 0
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
