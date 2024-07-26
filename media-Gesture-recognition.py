import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Load the pre-trained MediaPipe hand detection model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize variables
prev_num_fingers = 0
play_pause_executed = False
next_song_executed = False
prev_song_executed = False

# Helper function to count fingers
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    count = 0

    # Get the normalized landmark positions
    h, w, _ = frame.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    # Check the thumb
    if landmarks[thumb_tip][0] < landmarks[thumb_tip - 2][0]:
        count += 1

    # Check the other fingers
    for tip in finger_tips:
        if landmarks[tip][1] < landmarks[tip - 2][1]:
            count += 1

    return count

# Helper function to check for "okay" gesture
def is_okay_gesture(hand_landmarks):
    h, w, _ = frame.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    # Check distance between thumb tip and index finger tip
    thumb_tip = landmarks[4]
    index_finger_tip = landmarks[8]
    distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_finger_tip))

    return distance < 30  # You may need to adjust this threshold

# Helper function to check for thumbs up gesture
def is_thumbs_up_gesture(hand_landmarks):
    h, w, _ = frame.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    # Thumb tip should be above the base of the thumb
    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]

    # All other fingers should be curled (below the midpoint of the respective base and tip)
    finger_tips = [8, 12, 16, 20]
    curled = all(landmarks[tip][1] > landmarks[tip - 2][1] for tip in finger_tips)

    return thumb_tip[1] < thumb_base[1] and curled

# Helper function to check for victory gesture
def is_victory_gesture(hand_landmarks):
    h, w, _ = frame.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    # Index and middle fingers should be up, other fingers should be down
    index_finger_up = landmarks[8][1] < landmarks[6][1]
    middle_finger_up = landmarks[12][1] < landmarks[10][1]
    ring_finger_down = landmarks[16][1] > landmarks[14][1]
    pinky_finger_down = landmarks[20][1] > landmarks[18][1]

    return index_finger_up and middle_finger_up and ring_finger_down and pinky_finger_down

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe hands model
    results = hands.process(frame_rgb)

    # If hand(s) detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of visible fingers
            num_fingers = count_fingers(hand_landmarks)

            # Check for "okay" gesture
            if is_okay_gesture(hand_landmarks):
                if not play_pause_executed:
                    pyautogui.press('playpause')
                    play_pause_executed = True
            else:
                play_pause_executed = False

            # Check for thumbs up gesture
            if is_thumbs_up_gesture(hand_landmarks):
                if not next_song_executed:
                    pyautogui.press('nexttrack')
                    next_song_executed = True
            else:
                next_song_executed = False

            # Check for victory gesture
            if is_victory_gesture(hand_landmarks):
                if not prev_song_executed:
                    pyautogui.press('prevtrack')
                    prev_song_executed = True
            else:
                prev_song_executed = False

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
