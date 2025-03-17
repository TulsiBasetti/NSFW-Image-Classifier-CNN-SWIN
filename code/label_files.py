import os
import pandas as pd

# Define paths to dataset folders
train_images_dir = r"C:\6 SEMESTER\Minor_Project\dataset\train"
val_images_dir = r"C:\6 SEMESTER\Minor_Project\dataset\validate"
test_images_dir = r"C:\6 SEMESTER\Minor_Project\dataset\test"
labels_dir = r"C:\6 SEMESTER\Minor_Project\dataset\labels"  # Folder to save label files

# Class IDs for explicit content (from your .py file)
explicit_classes = [0, 1, 2, 3, 4]  # Dick (0), Vagina (1), Breast (2), Butt (3), genital_area (4)

# Function to determine nudity level
def get_nudity_level(label_file):
    """
    Check if a label file contains any explicit classes and determine the nudity level.
    """
    if not os.path.exists(label_file):
        return 'regular'  # If no label file exists, assume the image is safe
    with open(label_file, 'r') as f:
        explicit_count = 0
        for line in f:
            class_id = int(line.split()[0])  # First value in the line is the class ID
            if class_id in explicit_classes:
                explicit_count += 1
    if explicit_count == 0:
        return 'regular'
    elif explicit_count <= 2:
        return 'semi-nudity'
    else:
        return 'full-nudity'

# Function to create image-level labels
def create_image_labels(images_dir, labels_dir):
    """
    Create a dictionary of image-level labels ("regular", "semi-nudity", "full-nudity") for all images in a folder.
    """
    image_labels = {}
    for image_name in os.listdir(images_dir):
        if image_name.endswith('.jpg'):  # Assuming images are in .png format
            label_file = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))
            image_labels[image_name] = get_nudity_level(label_file)
    return image_labels

# Generate labels for train, val, and test sets
train_labels = create_image_labels(train_images_dir, labels_dir)
val_labels = create_image_labels(val_images_dir, labels_dir)
test_labels = create_image_labels(test_images_dir, labels_dir)

# Save labels to CSV files
def save_labels_to_csv(labels, csv_path):
    """
    Save image-level labels to a CSV file.
    """
    df = pd.DataFrame(list(labels.items()), columns=['image', 'label'])
    df.to_csv(csv_path, index=False)

save_labels_to_csv(train_labels, r'C:\6 SEMESTER\Minor_Project\dataset\train_labels.csv')
save_labels_to_csv(val_labels, r'C:\6 SEMESTER\Minor_Project\dataset\val_labels.csv')
save_labels_to_csv(test_labels, r'C:\6 SEMESTER\Minor_Project\dataset\test_labels.csv')

print("Image-level labels generated and saved to CSV files.")


