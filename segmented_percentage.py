import json
from collections import defaultdict
import os
from PIL import Image

def generate_class_percentages(image_folder, json_file, output_file):
    # Step 1: Read the JSON file
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    class_color_mapping = {
        label['readable']: tuple(label['color'])
        for label in json_data['labels']
    }

    result = {}

    # Step 2-6: Process each image
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Step 2: Read segmented image
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)

            # Step 3: Classify pixels
            pixels = image.load()
            width, height = image.size
            class_counts = defaultdict(int)
            total_pixels = 0
            for y in range(height):
                for x in range(width):
                    pixel_color = tuple(pixels[x, y])
                    for class_name, class_color in class_color_mapping.items():
                        if pixel_color == class_color:
                            # Step 4: Count pixel occurrences
                            class_counts[class_name] += 1
                            total_pixels += 1
                            break

            # Step 5: Calculate percentages
            class_percentages = {
                class_name: count / total_pixels * 100
                for class_name, count in class_counts.items()
            }

            # Step 6: Store results with image title
            result[filename] = {
                'class_percentages': class_percentages,
                'image_path': image_path
            }

            print(f"'{filename}' percentages saved successfully.")

    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)


    print(f"done!")

# Usage example
image_folder = '/home/darendy/MaskFormer_ADE20K/segmented_images'
json_file = '/home/darendy/MaskFormer_ADE20K/objectName150_colors150.json'
output_file = '/home/darendy/MaskFormer_ADE20K/percentages_json/output_file.json'
generate_class_percentages(image_folder, json_file, output_file)