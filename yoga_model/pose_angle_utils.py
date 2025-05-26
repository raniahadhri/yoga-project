# Pose landmark extraction with MediaPipe

from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from mediapipe.python.solutions.pose import PoseLandmark
import random

#model = load_model('models/yoga_pose_classifier.h5')



def calculate_angle(a, b, c):
    """
    Calculates angle (in degrees) between three points:
    a, b, c are (x, y) tuples. b is the vertex.
    """
    a = [a[0], a[1]]
    b = [b[0], b[1]]
    c = [c[0], c[1]]

    # Calculate vectors
    ab = [a[0] - b[0], a[1] - b[1]]
    cb = [c[0] - b[0], c[1] - b[1]]

    # Dot product and magnitude
    dot_product = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])

    # Angle in radians -> degrees
    angle = math.acos(dot_product / (mag_ab * mag_cb + 1e-6))
    return math.degrees(angle)

def load_image(image_path):
    image = cv2.imread(image_path)
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def detect_landmarks(image_rgb):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(image_rgb)
    return results

def extract_needed_landmarks(results, landmark_names):
    if not results.pose_landmarks:
        return {}
    lm = results.pose_landmarks.landmark
    needed = {}
    for name in landmark_names:
        try:
            landmark_enum = getattr(PoseLandmark, name)
            needed[name] = lm[landmark_enum]
        except AttributeError:
            print(f"Warning: '{name}' is not a valid PoseLandmark")
    return needed

def show_image(image_with_lms):
    plt.imshow(cv2.cvtColor(image_with_lms, cv2.COLOR_BGR2RGB))  
    plt.axis('off')
    plt.title("Needed Landmarks Only")
    plt.show()


def calculate_pose_angles(landmarks_dict, angle_defs):
    get_coords = lambda lm: (lm.x, lm.y)
    results = {}

    for angle_name, p1, vertex, p2 in angle_defs:
        if all(k in landmarks_dict for k in [p1, vertex, p2]):
            a = get_coords(landmarks_dict[p1])
            b = get_coords(landmarks_dict[vertex])
            c = get_coords(landmarks_dict[p2])
            angle = calculate_angle(a, b, c)
            results[angle_name] = angle
        else:
            results[angle_name] = None  # Or raise an error/log warning
    return results

def compute_pose_angles_from_image(image_path,angle_definitions,landmark_list): #general function
    try:
        image = load_image(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
    results = detect_landmarks(image)
    if results.pose_landmarks:
        needed_lms = extract_needed_landmarks(results, landmark_list)
    pose_angles = calculate_pose_angles(needed_lms, angle_definitions)
    return pose_angles

def compute_pose_angles_from_image(image_path, angle_definitions, landmark_list):
    image = load_image(image_path)
    results = detect_landmarks(image)
    if results.pose_landmarks:
        needed_lms = extract_needed_landmarks(results, landmark_list)
        pose_angles = calculate_pose_angles(needed_lms, angle_definitions)
        return pose_angles, needed_lms
    else:
        # Return empty dicts if no pose detected
        return {}, {}
    

def process_pose_images_in_folder(folder_path, angle_definitions, landmark_list):
    all_data = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # only images
            image_path = os.path.join(folder_path, filename)

            try:
                # Compute angles for the image
                angles, needed_lms = compute_pose_angles_from_image(image_path, angle_definitions, landmark_list)
                
                # Add filename info
                angles['image_name'] = filename
                all_data.append(angles)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Create a DataFrame
    df = pd.DataFrame(all_data)
    return df


def save_calculated_angles_in_csv(pose, csv_path, df_angles, angle_names):
    """
    Saves the mean and std deviation of selected angles for a pose to a CSV.

    Parameters:
    - pose (str): Name of the pose 
    - csv_path (str): Full path to the CSV file to save
    - df_angles (pd.DataFrame): DataFrame containing the angle values
    - angle_names (list of str): List of angle column names to process
    """

    mean_angles = df_angles[angle_names].mean()
    std_angles = df_angles[angle_names].std()

    rows = [
        {"pose": pose, "angle_name": angle, "angle_mean": mean, "angle_std": std}
        for (angle, mean), (_, std) in zip(mean_angles.items(), std_angles.items())
    ]

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)

    print(f"✅ CSV saved to: {csv_path}")
    return summary_df
