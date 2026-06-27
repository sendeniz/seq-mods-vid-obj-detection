import os
import pandas as pd

csv_file_path = 'data/ILSVRC2015/Data/val_max_80_frames.csv' 
df = pd.read_csv(csv_file_path, header=None, names=['ID'])

# Define the directory where the CSV files are located
base_dir = 'data/ILSVRC2015/Data/labels/val_frame_ids'  # Update with the directory path

# Get the list of .csv file names (without the .csv extension)
csv_files = [os.path.splitext(f)[0] for f in os.listdir(base_dir) if f.endswith('.csv')]

# Filter the DataFrame to keep only IDs that have corresponding .csv files in the directory
filtered_df = df[df['ID'].isin(csv_files)]

# Save the filtered DataFrame back to a CSV file
output_csv_file_path = 'data/ILSVRC2015/Data/val_max_80_frames.csv'
filtered_df.to_csv(output_csv_file_path, index=False, header=False)

print(f"Filtered IDs saved to {output_csv_file_path}")