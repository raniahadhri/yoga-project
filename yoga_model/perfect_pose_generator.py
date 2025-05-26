
from pose_angle_utils import process_pose_images_in_folder
import pandas as pd


landmark_list = [
    'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_ANKLE',
    'LEFT_KNEE', 'LEFT_ELBOW', 'LEFT_WRIST'
]

# Define which angles to calculate
angle_definitions_downdog = [
    ("Shoulder-Hip-Ankle", "LEFT_SHOULDER", "LEFT_HIP", "LEFT_ANKLE"),
    ("Hip-Knee-Ankle", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("Wrist-Elbow-Shoulder", "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER")
]
angle_names = ["Shoulder-Hip-Ankle", "Hip-Knee-Ankle", "Wrist-Elbow-Shoulder"]
folder = "C:/Users/hadhr/Documents/yoga-project/data/DATASET/Perfect_pose/downdog"
pose_name = "downdog"
csv_file = "angles_stats.csv"
df_angles = process_pose_images_in_folder(folder, angle_definitions_downdog, landmark_list)



def save_calculated_angles_in_csv(pose, csv_path, df_angles, angle_names):
    """
    Saves the mean and std deviation of selected angles for a pose to a CSV.

    Parameters:
    - pose (str): Name of the pose (e.g., 'downdog')
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

save_calculated_angles_in_csv(pose_name, csv_file, df_angles, angle_names)
