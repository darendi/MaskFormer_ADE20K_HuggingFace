import os
import json
from PIL import Image, ImageDraw, ImageFont

# Path to the folder containing segmented images
folder_path = '/home/darendy/MaskFormer_ADE20K/segmented_images'

# Path to the JSON file containing class labels and colors
json_file = '/home/darendy/MaskFormer_ADE20K/objectName150_colors150.json'

# Output folder for saving the images with legends
output_folder = '/home/darendy/MaskFormer_ADE20K/images_w_legend/segmented_images'

# Load the class labels and colors from the JSON file
with open(json_file, 'r') as file:
    data = json.load(file)
    class_labels = {tuple(label['color']): label['readable'] for label in data['labels']}

# Iterate over the segmented images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load the segmented image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        # Create a new blank image for the legend
        legend_width = 150  # Adjust the width as per your needs
        legend_height = image.height
        legend_image = Image.new('RGB', (legend_width, legend_height), 'white')
        draw = ImageDraw.Draw(legend_image)

        # Define the font for label text
        font = ImageFont.load_default()

        # Get the unique classes present in the image
        unique_classes = set(image.getdata())
        
        # Iterate over the unique classes and their colors
        y_offset = 0
        for class_label in unique_classes:
            if class_label in class_labels:
                label_text = class_labels[class_label]

                # Draw the label text
                draw.text((5, y_offset), label_text, fill='black', font=font)

                # Draw a rectangle filled with the color
                draw.rectangle([(100, y_offset), (130, y_offset + 20)], fill=class_label)

                # Adjust the y offset for the next label
                y_offset += 25

        # Merge the legend image with the segmented image
        merged_image = Image.new('RGB', (image.width + legend_width, image.height))
        merged_image.paste(legend_image, (0, 0))
        merged_image.paste(image, (legend_width, 0))

        # Save the resulting image with the legend
        output_path = os.path.join(output_folder, filename)
        merged_image.save(output_path)

        print(f"Segmented image '{filename}' saved successfully.")

print(f"done!")

