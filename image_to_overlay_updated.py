import os
import cv2
import numpy as np

# Define the paths to the folders containing the images and segmented images
image_folder = '/home/darendy/images'
segmentation_folder = '/home/darendy/MaskFormer_ADE20K/segmented_images'

# Create an output folder to store the overlaid images
output_folder = '/home/darendy/MaskFormer_ADE20K/overlay_images'
os.makedirs(output_folder, exist_ok=True)

# Iterate over the images in the image folder
for image_name in os.listdir(image_folder):
    # Get the image file path and corresponding segmented image file path
    image_path = os.path.join(image_folder, image_name)
    segmentation_name = os.path.splitext(image_name)[0] + '.png'
    segmentation_path = os.path.join(segmentation_folder, segmentation_name)
    
    # Load the image and segmented image
    image = cv2.imread(image_path)
    segmentation = cv2.imread(segmentation_path)
    
    # Check if the images were loaded successfully
    if image is None or segmentation is None:
        print(f"Failed to load {image_name} or its corresponding segmented image.")
        continue
    
    # Resize the segmented image to match the dimensions of the original image
    segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
    
    # Apply an alpha blending to overlay the segmented image on top of the original image
    alpha = 0.6  # Adjust the alpha value for desired transparency
    overlaid = cv2.addWeighted(image, alpha, segmentation, 1 - alpha, 0)
    
    # Save the overlaid image to the output folder
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, overlaid)

    print(f"'{image_name}'_overlay saved successfully.")

print("Overlay process completed.")
