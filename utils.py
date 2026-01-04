import numpy as np

def get_head_tilt(face_lms):
    """Calculates head tilt based on nose and eye positions."""
    # Landmark IDs: Nose (1), Left Eye (33), Right Eye (263)
    nose = face_lms.landmark[1]
    left_eye = face_lms.landmark[33]
    right_eye = face_lms.landmark[263]

    # Calculate horizontal center between eyes
    eye_center_x = (left_eye.x + right_eye.x) / 2
    
    # Return the tilt value
    return nose.x - eye_center_x

def is_fist(hand_lms):
    """Checks if the hand is in a fist (all fingers down)."""
    # Landmark IDs for tips: Index(8), Middle(12), Ring(16), Pinky(20)
    # Compare tip Y to the joint below it (pip joint)
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    
    is_closed = True
    for tip, pip in zip(tips, pip_joints):
        if hand_lms.landmark[tip].y < hand_lms.landmark[pip].y:
            is_closed = False # At least one finger is up
    return is_closed