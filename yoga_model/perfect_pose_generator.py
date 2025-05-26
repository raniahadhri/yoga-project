
from pose_angle_utils import process_pose_images_in_folder,save_calculated_angles_in_csv
import pandas as pd

#-----------------------------------------TREE--------------------------
landmark_list_tree = [
    "LEFT_SHOULDER", "LEFT_HIP", "LEFT_ANKLE",
    "LEFT_KNEE"
]
angle_definitions_tree = [
    ("Hip-Knee-Ankle", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),          # Standing leg angle
    ("Shoulder-Hip-Ankle", "LEFT_SHOULDER", "LEFT_HIP", "LEFT_ANKLE"), # Body alignment
    ("Knee-Hip-Shoulder", "LEFT_KNEE", "LEFT_HIP", "LEFT_SHOULDER")    # Trunk uprightness
]
angle_names_tree = ["Hip-Knee-Ankle", "Shoulder-Hip-Ankle", "Knee-Hip-Shoulder"]
folder_tree = "C:/Users/hadhr/Documents/yoga-project/data/DATASET/Perfect_pose/tree"
pose_name = "tree"

#-----------------------------------------downdog--------------------------
landmark_list_downdog = [
    'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_ANKLE',
    'LEFT_KNEE', 'LEFT_ELBOW', 'LEFT_WRIST'
]
angle_definitions_downdog = [
    ("Shoulder-Hip-Ankle", "LEFT_SHOULDER", "LEFT_HIP", "LEFT_ANKLE"),
    ("Hip-Knee-Ankle", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("Wrist-Elbow-Shoulder", "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER")
]
angle_names_downdog = ["Shoulder-Hip-Ankle", "Hip-Knee-Ankle", "Wrist-Elbow-Shoulder"]
folder_downdog = "C:/Users/hadhr/Documents/yoga-project/data/DATASET/Perfect_pose/downdog"
pose_name = "downdog"
#-------------------------------------------------------------------------------------------------

csv_file = "angles_stats.csv"


df_angles = process_pose_images_in_folder(folder_downdog, angle_definitions_downdog, landmark_list_downdog)
summary_df= save_calculated_angles_in_csv(pose_name, csv_file, df_angles, angle_names_downdog)
