import os
import csv

image_folder = "data/coco/images/"  
train_csv_file = "data/coco/train.csv" 
# we use val as test file, since test annotations are not publicly available
test_csv_file = "data/coco/test.csv" 

# Get all image filenames matching the pattern "train_id.jpg"
train_image_files = [filename for filename in os.listdir(image_folder) if filename.startswith("train") and filename.endswith(".jpg")]

test_image_files = [filename for filename in os.listdir(image_folder) if filename.startswith("val") and filename.endswith(".jpg")]

# Create a list of tuples containing (image_name, text_filename) pairs
train_data = [(image_name, image_name.replace(".jpg", ".txt")) for image_name in train_image_files]
test_data = [(image_name, image_name.replace(".jpg", ".txt")) for image_name in test_image_files]

# Write the data to a CSV file
with open(train_csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(train_data)
    
with open(test_csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(test_data)
    
