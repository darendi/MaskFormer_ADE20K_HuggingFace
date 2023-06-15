import os
import json
import torch
import cv2
from PIL import Image
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
import numpy as np


def overlay_segmentation(input_folder, segmented_folder, output_folder):
    # Load Mask2Former fine-tuned on Mapillary Vistas semantic segmentation
    processor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")

    # Load the JSON color map
    with open("/home/darendy/MaskFormer_ADE20K/objectName150_colors150.json") as f:
        color_map_data = json.load(f)

    # Create a mapping between class names and class indices
    class_name_to_index = {}
    for index, label in enumerate(color_map_data["labels"]):
        class_name_to_index[label["name"]] = index

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert("RGB")  # Convert to RGB format
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            # You can pass them to the processor for post-processing
            predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

            # Apply color mapping to the grayscale segmentation map
            colored_segmentation_map = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
            for label in color_map_data["labels"]:
                class_name = label["name"]
                class_index = class_name_to_index[class_name]
                color = label["color"]
                colored_segmentation_map[predicted_semantic_map == class_index] = color

            # Create a PIL Image from the colored segmentation map
            segmented_image = Image.fromarray(colored_segmentation_map)

            # Resize segmented image to match the original image size
            segmented_image = segmented_image.resize(image.size)

            # Convert PIL images to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv_segmented_image = cv2.cvtColor(np.array(segmented_image), cv2.COLOR_RGB2BGR)

            # Overlay segmented image over the original image
            alpha = 0.5  # Adjust transparency of the overlay
            overlay = cv2.addWeighted(cv_image, 1 - alpha, cv_segmented_image, alpha, 0)

            # Create output file path
            output_path = os.path.join(output_folder, filename)

            # Save the overlaid image
            cv2.imwrite(output_path, overlay)

            print(f"Processed image: {filename}, Saved as: {output_path}")

    print("Done!")


# Specify the input folder path containing the original images
input_images_folder = "/home/darendy/images"

# Specify the folder path containing the segmented images
segmented_images_folder = "/home/darendy/MaskFormer_ADE20K/segmented_images"

# Specify the output folder path for the overlaid images
output_overlay_folder = "/home/darendy/MaskFormer_ADE20K/overlay_images"

# Call the function to overlay the segmented images over the original images and save them in the output folder
overlay_segmentation(input_images_folder, segmented_images_folder, output_overlay_folder)


