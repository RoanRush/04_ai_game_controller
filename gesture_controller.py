import cv2
import mediapipe as mp
import pyautogui
from utils import get_head_tilt, is_fist # Import your own custom tools!

# Setup
cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process both Face and Hands
    face_results = face_mesh.process(img_rgb)
    hand_results = hands.process(img_rgb)

    # 1. Handle Steering (Face)
    if face_results.multi_face_landmarks:
        tilt = get_head_tilt(face_results.multi_face_landmarks[0])
        # Better steering for web games
        if tilt < -0.02:
            pyautogui.keyDown('left')
            pyautogui.keyUp('right') # Release the other key!
            cv2.putText(img, "<< LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif tilt > 0.02:
            pyautogui.keyDown('right')
            pyautogui.keyUp('left')
            cv2.putText(img, "RIGHT >>", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # If head is straight, release both
            pyautogui.keyUp('left')
            pyautogui.keyUp('right')
            
    # 2. Handle Brake/Stop (Hands)
    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            if is_fist(hand_lms):
                pyautogui.press('space') # Map 'space' to the brake
                cv2.putText(img, "BRAKE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("RTX 5060 AI Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()