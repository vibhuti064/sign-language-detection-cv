import os
import pickle

# Logging suppression (Good practice)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import mediapipe 
import cv2
import matplotlib.pyplot as plt



mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
# Setting static_image_mode=False is correct for batch processing
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5) 


DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # data_aux, x_, y_ initialization should happen here if you process multiple times, 
        # but since they are initialized inside the conditional block, it's safe.
        
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        # Check if image load failed (e.g., corrupt file)
        if img is None:
            continue
        img = cv2.flip(img, 1)
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        # Removed: final_data_aux = [] (Unused variable)
        
        if results.multi_hand_landmarks:
            
            # --- 1. SINGLE HAND ENFORCEMENT ---
            # Process ONLY the first detected hand (index 0)
            hand_landmarks = results.multi_hand_landmarks[0]
            # --- -------------------------- ---

            data_aux = []
            x_ = []
            y_ = []

            # 2. Collect all X and Y coordinates (to find min/max for normalization)
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Check if lists are empty (highly unlikely if detection succeeded, but safe)
            if not x_:
                continue

            min_x = min(x_)
            min_y = min(y_)

            # 3. Normalize and append the data for this single hand (42 features total)
            for landmark in hand_landmarks.landmark:
                # Append normalized (x - min(x)) and (y - min(y))
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # 4. Append the 42-feature sample and the label
            data.append(data_aux)
            labels.append(dir_) 
        # Removed: 'else: pass' comment block for cleaner code


f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print(f"Data processing complete. Total samples saved: {len(data)}")