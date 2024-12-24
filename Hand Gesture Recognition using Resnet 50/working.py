import cv2
import mediapipe as mp
import numpy as np
import os
from keras.models import load_model
import joblib
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

script_dir = os.path.dirname(os.path.abspath(__file__))
label_encoder_path = 'label_encoder_resnet.joblib'
model_path = 'best_model_fold_2.h5'

label_encoder = joblib.load(label_encoder_path)
model = load_model(model_path)

last_prediction_time = 0
prediction_interval = 1  # Interval in seconds for making predictions

box_size = 150

# State flag to determine if Gesture 0 has been detected
gesture_0_detected = False

# Variable to handle initial hand detection delay
initial_frame_count = 0
frame_stabilization_threshold = 10  # Number of frames for stabilization

confidence_threshold = 0.8  # Set a threshold for prediction confidence
gesture_0_stability_count = 0  # To count frames where Gesture 0 is detected

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    landmarks_List = []
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks_List.append([i, x, y])

    if landmarks_List:
        # Calculate the center of the hand
        hand_center_x = sum(landmark[1] for landmark in landmarks_List) // len(landmarks_List)
        hand_center_y = sum(landmark[2] for landmark in landmarks_List) // len(landmarks_List)

        # Calculate the bounding box coordinates around the hand
        x1 = max(0, hand_center_x - int(box_size / 2))
        y1 = max(0, hand_center_y - int(box_size / 2))
        x2 = min(w, hand_center_x + int(box_size / 2))
        y2 = min(h, hand_center_y + int(box_size / 2))

        # Draw the blue box around the hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check if it's time to make a prediction (every 1 second)
        if (time.time() - last_prediction_time) >= prediction_interval:
            # Extract ROI from the blue box
            roi = frame[y1 + 2:y2 - 2, x1 + 2:x2 - 2]
            resized_roi = cv2.resize(roi, (128, 128))

            img = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(img, axis=0)

            # Perform gesture recognition
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            prediction_confidence = np.max(predictions)  # Get the confidence of the predicted class

            # Skip initial frames for Gesture 0 detection (avoid false prediction on initial hand introduction)
            if initial_frame_count < frame_stabilization_threshold:
                initial_frame_count += 1
                decoded_predictions = "Waiting for gesture recognition..."
            else:
                # Apply a confidence threshold to avoid false positives
                if prediction_confidence >= confidence_threshold:
                    if not gesture_0_detected:
                        # Only predict Gesture 0 initially
                        if predicted_class == 0:
                            decoded_predictions = label_encoder.inverse_transform(predicted_class.ravel())[0]
                            print(f'Predicted Gesture: {decoded_predictions}')
                            gesture_0_detected = True
                            gesture_0_stability_count = 0  # Reset the stability count
                            # Set a short delay before the next prediction to avoid repeating the gesture
                            last_prediction_time = time.time()
                    else:
                        # After Gesture 0 is detected, allow predictions from 1 to 13
                        if 1 <= predicted_class <= 13:
                            decoded_predictions = label_encoder.inverse_transform(predicted_class.ravel())[0]
                            print(f'Predicted Gesture: {decoded_predictions}')
                            # Reset back to only recognizing Gesture 0 after a prediction from 1 to 13
                            gesture_0_detected = False
                            last_prediction_time = time.time()
                            gesture_0_stability_count = 0  # Reset stability count
                else:
                    decoded_predictions = "Low Confidence"
                    # If confidence is low, count the number of frames with Gesture 0 for stability
                    if predicted_class == 0:
                        gesture_0_stability_count += 1
                    else:
                        gesture_0_stability_count = 0  # Reset if another gesture is detected

                    # Only detect Gesture 0 after it has been consistently detected for a few frames
                    if gesture_0_stability_count > 5:
                        gesture_0_detected = True
                        print("Gesture 0 detected consistently.")
                        last_prediction_time = time.time()

    else:
        decoded_predictions = "No hand detected"
        print(decoded_predictions)

    # Show the frame with hand tracking
    cv2.imshow('Hand Tracking', frame)

    # Print prediction information on the frame
    cv2.putText(frame, f"Prediction: {decoded_predictions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
