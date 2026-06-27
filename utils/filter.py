import os

# Path to the folder containing the text file with video IDs
text_file_folder = "data/ILSVRC2015/Data/labels/"

# Path to the folder containing the CSV files
csv_file_folder = "data/ILSVRC2015/Data/labels/val_frame_ids/"

# Read video IDs from the text file
with open(os.path.join(text_file_folder, "val_vid_id.txt"), "r") as file:
    video_ids = file.read().splitlines()

# Check which video IDs have corresponding CSV files
valid_video_ids = []
for video_id in video_ids:
    csv_file_path = os.path.join(csv_file_folder, f"{video_id}.csv")
    if os.path.exists(csv_file_path):
        valid_video_ids.append(video_id)

# Write the filtered list of video IDs back to the text file
with open(os.path.join(text_file_folder, "val_vid_id.txt"), "w") as file:
    file.write("\n".join(valid_video_ids))