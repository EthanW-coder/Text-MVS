import torch
import os
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# Set device
device_id = 0
model_dir = "Qwen2.5-VL-7B-Instruct"  # Model directory

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map=f"cuda:{device_id}"
)

# Load processor
processor = AutoProcessor.from_pretrained(model_dir, max_pixels=1280 * 28 * 28)


def process_single_image(image_path):
    """Process a single image and generate description"""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            return None

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text",
                     "text": "Describe this aerial image in one sentence, outputting the scale and depth of each visible object, as well as the appearance details between them and their absolute positions in the overall image."},
                ],
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if output_text else None

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def find_all_image_files(folder_path):
    """Recursively find all image files in a folder"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []

    # Recursively traverse all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(os.path.join(root, file))
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))

    return image_files


def check_label_exists(image_path, output_folder):
    """
    Check if an image already has a corresponding label file

    Args:
        image_path: Image file path
        output_folder: Output folder path

    Returns:
        bool: True if label file exists, False otherwise
    """
    # Get relative path from input folder
    rel_path = os.path.relpath(image_path, input_folder)

    # Generate corresponding label file path
    label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    label_path = os.path.join(output_folder, os.path.dirname(rel_path), label_filename)

    # Check if label file exists and is not empty
    return os.path.exists(label_path) and os.path.getsize(label_path) > 0


def batch_process_images(input_folder, output_folder=None):
    """
    Batch process all images in a folder

    Args:
        input_folder: Input image folder path
        output_folder: Output text folder path, if None use input folder
    """
    # Set output folder
    if output_folder is None:
        output_folder = input_folder

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files (including subdirectories)
    image_files = find_all_image_files(input_folder)

    print(f"Found {len(image_files)} images to process")

    if not image_files:
        print("No image files found, please check if the path is correct")
        print(f"Search path: {input_folder}")
        return

    # Count skipped and processed images
    skipped_count = 0
    processed_count = 0

    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Checking ({i + 1}/{len(image_files)}): {image_path}")

        # Check if label file already exists
        if check_label_exists(image_path, output_folder):
            print(f"Label already exists, skipping: {image_path}")
            skipped_count += 1
            continue

        print(f"Processing ({i + 1 - skipped_count}/{len(image_files) - skipped_count}): {image_path}")

        # Generate description
        description = process_single_image(image_path)

        if description:
            # Create corresponding output directory structure
            rel_path = os.path.relpath(image_path, input_folder)
            output_subdir = os.path.dirname(os.path.join(output_folder, rel_path))
            os.makedirs(output_subdir, exist_ok=True)

            # Generate output file path (keep original filename, change extension to .txt)
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            output_path = os.path.join(output_subdir, output_filename)

            # Write description to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(description)

            print(f"Generated: {output_path}")
            processed_count += 1
        else:
            print(f"Processing failed: {image_path}")

        print("-" * 50)

    # Output statistics
    print(f"\nProcessing completed!")
    print(f"Total images: {len(image_files)}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process: {len(image_files) - skipped_count - processed_count}")


if __name__ == "__main__":
    # Set input folder path
    input_folder = ""

    output_folder = None

    # Execute batch processing
    batch_process_images(input_folder, output_folder)