
import argparse
from utils.comm_util import get_images, scale_image_to_folder
from ImageClassifier import ImageClassifier
from PhotoQualityOrganizer import PhotoQualityOrganizer
from photo_comparer import find_similar_groups

def main():
    # Parse command line arguments for input and output paths
    parser = argparse.ArgumentParser(description="AI PhotoSorter")
    parser.add_argument('input_path', type=str, help='Path to the folder containing images')
    parser.add_argument('--output_path', type=str, help='Path to save processed images', required=True)
    parser.add_argument('--sort_order', type=str, choices=['ASC', 'DESC'], help='Sort order for images', default='DESC')
    args = parser.parse_args()

    # Initialize the image classifier and quality organizer
    classifier = ImageClassifier()
    photo_quality_organizer = PhotoQualityOrganizer()

    # Load images from the specified path
    image_list = get_images(args.input_path, include_subfolders=True)
    
    for image_path in image_list:
        # Detect the category of the image
        class_name = classifier.detect_image_category(image_path)
        print(f"Image Path: {image_path}, Class Name: {class_name}")

        # Calculate the quality score of the image
        quality_score = photo_quality_organizer.calculate_score(image_path)
        print(f"Image Path: {image_path}, Quality Score: {quality_score}")

    # Sorting and filtering images
    is_desc = args.sort_order != 'ASC'
    photos = photo_quality_organizer.get_photos_in_path(args.input_path, 20, is_desc)
    photo_paths = [item[0] for item in photos]

    # Discard similar photos, keep only the best ones
    similar_groups, remaining_images = find_similar_groups(photo_paths, 0.9)
    unique_image_paths = remaining_images
    for group in similar_groups:
        group.sort(key=photo_quality_organizer.calculate_score, reverse=True)
        unique_image_paths.append(group[0])
    
    # Scale and save the unique images to the output folder
    for image_path in unique_image_paths:
        print(f'{image_path}: {photo_quality_organizer.calculate_score(image_path)}')
        scale_image_to_folder(image_path, args.output_path)

if __name__ == "__main__":
    main()
