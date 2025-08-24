import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def remove_background(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray_image, 245, 255, cv2.THRESH_BINARY_INV) 

    if image.shape[2] == 4:  #if img has an alpha channel 
        b, g, r, _ = cv2.split(image)  # ignore the alpha channel
    elif image.shape[2] == 3:  # If in BGR format
        b, g, r = cv2.split(image)
    else:
        print(f"Unexpected number of channels: {image.shape[2]}")
        return None
    rgba_image = cv2.merge((b, g, r, mask))

    return rgba_image

def overlay_transparent(background, overlay, x, y, overlay_width, overlay_height):
    overlay = cv2.resize(overlay, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]

    if x >= background.shape[1] or y >= background.shape[0]:
        return background
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]

    roi = background[y:y+h, x:x+w]

    if overlay.shape[2] == 4:  # Check if overlay has an alpha channel
        overlay_img = overlay[..., :3]  # Extract RGB channels
        mask = overlay[..., 3:] / 255.0  # Extract alpha channel and normalize
        roi = (1.0 - mask) * roi + mask * overlay_img
        background[y:y+h, x:x+w] = roi
    return background

def overlay_clothing(clothing_img, frame, landmarks):
    # Get shoulder and hip landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

    # Calculate shoulder width and body height
    shoulder_width = int(np.linalg.norm([right_shoulder.x - left_shoulder.x, 
                                         right_shoulder.y - left_shoulder.y]) * frame.shape[1])
    torso_height = int(np.linalg.norm([left_shoulder.y - left_hip.y, 
                                       left_shoulder.x - left_hip.x]) * frame.shape[0])

    #Increase scaling factors to make the clothing appear larger
    scaling_factor = 1.63 
    shoulder_width = int(shoulder_width * scaling_factor)
    torso_height = int(torso_height * scaling_factor)

    #average shoulder position  for placing the clothing
    avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    avg_shoulder_y = (left_shoulder.y + left_shoulder.y) / 2

    #adjust the Y position to align with the shoulders more than the hips
    overlay_position_x = int(avg_shoulder_x * frame.shape[1] - shoulder_width // 2)
    overlay_position_y = int(avg_shoulder_y * frame.shape[0]) - int(torso_height * 0.221)  #lower value for crop tops

    #resize clothing image to match shoulder width and body height
    frame = overlay_transparent(frame, clothing_img, overlay_position_x, overlay_position_y, shoulder_width, torso_height)

    return frame


def start_webcam_with_overlay(clothing_path):

    clothing_img = remove_background(clothing_path)
    if clothing_img is None:
        raise ValueError("Clothing image not loaded or processed")
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        result = pose.process(frame_rgb)

        # Check if landmarks are detected
        if result.pose_landmarks:
            # Overlay clothing on the frame using landmarks
            frame = overlay_clothing(clothing_img, frame, result.pose_landmarks.landmark)

        cv2.imshow('Webcam Try-On', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path='C:/Users/Shreya Sameer/Desktop/fcv'
    abs_path = path+'/static/cloth/09715_00.jpg' 
    start_webcam_with_overlay(abs_path)  