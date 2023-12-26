import argparse
from PIL import Image
from database_manager import DatabaseManager
from utils.comm_util import *
from utils.file_md5 import get_image_md5
from logger_unit import ConsoleColor, CustomLogger
from utils.multimedia_helper import get_photo_create_time


def compare_similarity(image1, image2):
    if image1 is None or image2 is None:
        return None
    return ImageSimilarityComparer.get_instance().compare(image1 , image2)

def find_similar_groups(image_paths, similarity_threshold = 0.8):
    return ImageSimilarityComparer.get_instance().find_similar_groups(image_paths, similarity_threshold)


class ImageSimilarityComparer:

    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = None
        self.lpips_model = None
        self.transform = None
        self.image_cache = {} # Used for caching image features
        self.db_manager = DatabaseManager.get_instance()
        self.logger = CustomLogger('ImageSimilarityComparer')

    def init_ai(self):
        if self.device is None:

            import lpips
            import torch
            from torchvision import models, transforms

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)  # Using LPIPS with AlexNet
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def clear_cache(self):
        self.image_cache.clear()

    # Load and transform images with caching
    def transform_image(self, image_path):
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        image = Image.open(image_path).convert('RGB')
        transformed_image = self.transform(image).to(self.device)
        self.image_cache[image_path] = transformed_image
        return transformed_image

    def compare(self, image1, image2):
        md5_1 = get_image_md5(image1)
        md5_2 = get_image_md5(image2)
        similarity = self.db_manager.load_similarity(md5_1, md5_2)
        if similarity is not None and similarity > 0:
            return similarity

        self.init_ai()
        try:
            image1_tensor = self.transform_image(image1)
            image2_tensor = self.transform_image(image2)
            similarity = self.lpips_model(image1_tensor, image2_tensor).item()
            similarity = max(0, min(1, similarity))
            similarity = 1 - similarity
            self.db_manager.save_similarity(md5_1, md5_2, similarity)
            return similarity
        except RuntimeError as e:
            self.logger.error(f'compare error: {e}, image1: {image1}, image2: {image2}')
            self.clear_cache()
            return None

    # Image similarity grouping; due to slow computation, pairwise comparison of all images is impractical, so sort by time and compare sequentially; exit if more than 5 different photos are found, assuming similar photos are taken at the same time
    def find_similar_groups(self, image_paths, similarity_threshold = 0.8):
        similar_groups = []
        remaining_images = [] 
        similar_images = set()

        image_with_time_list = [(path, get_photo_create_time(path)) for path in image_paths]
        image_with_time_list.sort(key=lambda x: x[1])
        n = len(image_with_time_list)
        temp_similar = set()
        for i in range(n):
            diff_count = 0
            image1 = image_with_time_list[i][0]
            time1 = image_with_time_list[i][1]

            if image1 in similar_images:                
                continue

            for j in range(i + 1, n):                
                image2 = image_with_time_list[j][0]                
                time2 = image_with_time_list[j][1]
                if image2 in similar_images:
                    continue
                time_diff = abs(time1 - time2)
                if time_diff > 5 * 60 * 60: # 5 hours
                    break
                similarity = self.compare(image1, image2)
                if similarity and similarity >= similarity_threshold:
                    self.logger.info(f"find_similar_groups,found similar photo: {image1} {image2} {similarity}", ConsoleColor.YELLOW) 
                    temp_similar.update([image1, image2])                    
                else:
                    diff_count +=1

                if diff_count > 5:
                    break


            if len(temp_similar) > 1:
                similar_groups.append(list(temp_similar))
            else:
                remaining_images.append(image_with_time_list[i][0])

            similar_images.update(temp_similar)
            temp_similar.clear()

        return similar_groups, remaining_images

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, help='Command to execute')
    parser.add_argument('--input_path', type=str, default='E:\\test\\', help='Path to input file')
    parser.add_argument('--output_path', type=str, default='E:\\output\\', help='Path to output file')    
    args,_ = parser.parse_known_args()  
    print(args)

    image_flies = get_images(args.input_path, True)
    for i in range(len(image_flies)):
        for j in range(i+1, len(image_flies)):
            image1 = image_flies[i]
            image2 = image_flies[j]
            similarity = compare_similarity(image1, image2)
            if similarity > 0.8:
                CustomLogger.print(f'{image1} vs {image2}: {similarity}', ConsoleColor.RED)
            elif similarity > 0.7:
                CustomLogger.print(f'{image1} vs {image2}: {similarity}', ConsoleColor.YELLOW)
            elif similarity > 0.65:
                CustomLogger.print(f'{image1} vs {image2}: {similarity}', ConsoleColor.BLUE)
            elif similarity > 0.6:
                CustomLogger.print(f'{image1} vs {image2}: {similarity}', ConsoleColor.GREEN)
            else:
                pass
                # CustomLogger.print(f'{image1} vs {image2}: {similarity}', ConsoleColor.GREEN)     
    