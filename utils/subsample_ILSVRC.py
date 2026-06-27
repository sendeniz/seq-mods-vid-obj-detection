import os
import pandas as pd

max_length = 80

# Load the CSV file
csv_file_path = 'data/ILSVRC2015/Data/train.csv'  # Update with your CSV file path
df = pd.read_csv(csv_file_path, header=None)
# Extract the IDs from the CSV file
folder_ids = df[0].tolist()  # Replace 'id_column_name' with the actual column name in your CSV

# Define the base directory where the folders are located
base_dir = 'data/ILSVRC2015/Data/VID/train/'  # Update with the base directory path

# Initialize counters
total_folders = 0
folders_leq_items = 0
small_folders = []

# Loop through each folder ID
for folder_id in folder_ids:
    folder_path = os.path.join(base_dir, folder_id)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        total_folders += 1
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        item_count = len(image_files)
        if item_count <= max_length:
            folders_leq_items += 1
            small_folders.append(folder_id)

# Print the results
print(f"Total number of folders: {total_folders}")
print(f"Number of folders with <= {max_length} items: {folders_leq_items}")
print(f"Folders with <= {max_length} items: {len(small_folders)}")

# Save the small_folders to a CSV file
small_folders_df = pd.DataFrame(small_folders, columns=["Folder_ID"])
output_csv_file_path = f'train_max_{max_length}_frames.csv'
small_folders_df.to_csv(output_csv_file_path, index=False)

print(f"Small folders saved to {output_csv_file_path}")


# Load the CSV file
csv_file_path = 'data/ILSVRC2015/Data/val.csv'  # Update with your CSV file path
df = pd.read_csv(csv_file_path, header=None)
# Extract the IDs from the CSV file
folder_ids = df[0].tolist()  # Replace 'id_column_name' with the actual column name in your CSV

# Define the base directory where the folders are located
base_dir = 'data/ILSVRC2015/Data/VID/val/'  # Update with the base directory path

# Initialize counters
total_folders = 0
folders_leq_items = 0
small_folders = []

# Loop through each folder ID
for folder_id in folder_ids:
    folder_path = os.path.join(base_dir, folder_id)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        total_folders += 1
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        item_count = len(image_files)
        if item_count <= max_length:
            folders_leq_items += 1
            small_folders.append(folder_id)

# Print the results
print(f"Total number of folders: {total_folders}")
print(f"Number of folders with <= {max_length} items: {folders_leq_items}")
print(f"Folders with <= {max_length} items: {len(small_folders)}")

# Save the small_folders to a CSV file
small_folders_df = pd.DataFrame(small_folders, columns=["Folder_ID"])
output_csv_file_path = f'val_max_{max_length}_frames.csv'
small_folders_df.to_csv(output_csv_file_path, index=False)

print(f"Small folders saved to {output_csv_file_path}")