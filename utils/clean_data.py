import csv
import os

# Path to your CSV file
csv_file_path = 'data/coco/train_10Kexamples.csv'

# Read the CSV file and filter out entries with missing files
filtered_entries = []
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        _, txt_file = row
        txt_path = os.path.join('data/coco/labels/', txt_file)

        # Check if the txt file exists
        if os.path.exists(txt_path):
            filtered_entries.append(row)

# Write the filtered entries back to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(filtered_entries)


# Path to your CSV file
csv_file_path = 'data/coco/test_416examples.csv'

# Read the CSV file and filter out entries with missing files
filtered_entries = []
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        _, txt_file = row
        txt_path = os.path.join('data/coco/labels/', txt_file)

        # Check if the txt file exists
        if os.path.exists(txt_path):
            filtered_entries.append(row)

# Write the filtered entries back to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(filtered_entries)
