import argparse
from PIL import Image

from database_manager import DatabaseManager
from logger_unit import CustomLogger
from photo_comparer import find_similar_groups
from utils.comm_util import *
from utils.file_md5 import get_image_md5
# import pyiqa.archs.topiq_swin as topiq_swin

#https://github.com/chaofengc/IQA-PyTorch/blob/main/tests/IAA_benchmark_results.csv

# 'topiq_iaa': {
#     'metric_opts': {
#         'type': 'CFANet',
#         'semantic_model_name': 'swin_base_patch4_window12_384',
#         'model_name': 'cfanet_iaa_ava_swin',
#         'use_ref': False,
#         'inter_dim': 512,
#         'num_heads': 8,
#         'num_class': 10,
#     },
#     'metric_mode': 'NR',
# },

metric_names = ['topiq_iaa','nima-vgg16-ava','nima','clipiqa','laion_aes','topiq_iaa_res50']


# Standard quality assessment
class PhotoQualityOrganizer:

    def __init__(self):
        self.logger = CustomLogger('PhotoQualityOrganizer')
        self.db_manager = DatabaseManager.get_instance()   
        self.metric_name = 'topiq_iaa'

        self.device = None
        self.transform = None
        self.iqa_metric = None

        self.score_cache = {}

    def __del__(self):
        if len(self.score_cache) > 0:
            self.db_manager.batch_save_score(self.score_cache, self.metric_name)
            self.score_cache = {}


    def init_ai(self):
        if self.device is None:
            import torch
            import pyiqa
            from torchvision import transforms
            # Deleted comments to help you load the model locally
            # pretrained_model_path = os.path.join(os.path.dirname(__file__), 'cfanet_iaa_ava_swin-393b41b4.pth')
            # swin_model_path = os.path.join(os.path.dirname(__file__), 'swin_base_patch4_window12_384_22kto1k.pth')

            # state_dict = torch.load(swin_model_path)
            # topiq_swin.default_cfgs['swin_base_patch4_window12_384']['state_dict'] = state_dict
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            
            # self.iqa_metric = pyiqa.create_metric(self.metric_name,pretrained_model_path = pretrained_model_path).to(self.device)
            self.iqa_metric = pyiqa.create_metric(self.metric_name).to(self.device)

            
            # self.iqa_metric.net.semantic_model.load_state_dict(state_dict)
            self.transform = transforms.Compose([
                transforms.Resize(384),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def calculate_score(self, image_path):
        self.init_ai()
        md5 = get_image_md5(image_path) 
        score = self.db_manager.load_score(md5, self.metric_name)
        if score:
            return score
        else:
            if md5 in self.score_cache:
                return self.score_cache[md5]
            try:  
                image = Image.open(image_path).convert('RGB')
                img_tensor = self.transform(image).to(self.device)               
                score = self.iqa_metric(img_tensor.unsqueeze(0)).item()
                self.score_cache[md5] = score
                # self.db_manager.save_score(file_md5, score, self.metric_name)
            except Exception as e:
                self.logger.error(f'error calculating score for {image_path}: {e}')
                score = 0
            
            # Batch caching
            if len(self.score_cache) > 500:
                self.db_manager.batch_save_score(self.score_cache, self.metric_name)
                self.score_cache = {}
            return score
        

    def get_photos_in_path(self, path = None, count = 10 , is_desc = True):
        if path is None:
            query = f'''SELECT p.file_path, s.score, s.metric_name
                    FROM photo_path p
                    INNER JOIN all_scores s ON p.md5 = s.md5
                    where s.metric_name = ?
                    ORDER BY s.score {'DESC' if is_desc else ''} 
                    LIMIT ?'''
            cursor = self.db_manager.conn.execute(query, (self.metric_name, count,))
        else:
            path = os.path.normpath(path)
            query = f'''SELECT p.file_path, s.score 
                    FROM photo_path p
                    INNER JOIN all_scores s ON p.md5 = s.md5
                    WHERE s.metric_name = ? and p.file_path LIKE ? 
                    ORDER BY s.score {'DESC' if is_desc else ''}  
                    LIMIT ?'''
            cursor = self.db_manager.conn.execute(query, (self.metric_name, f'%{path}%', count))

        return cursor.fetchall()    
 
    # Experimental function
    def diy_sql(self, query, params):
        cursor = self.db_manager.conn.execute(query, params)
        return cursor.fetchall()


# Usage example
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, help='Command to execute')
    parser.add_argument('--sort_order', type=str, default='DESC', help='DESC or ASC')
    parser.add_argument('--input_path', type=str, default='E:\\test\\', help='Path to input file')
    parser.add_argument('--output_path', type=str, default='E:\\output\\', help='Path to output file')    
    args,_ = parser.parse_known_args()  
    print(args)

    photo_quality_organizer = PhotoQualityOrganizer()

    # Calculate photo quality, a mandatory step; without it, subsequent operations are just database queries and won't result in photo sorting
    if args.command is None or args.command == 'calculate':
        image_flies = get_images(args.input_path, True)        
        for image_path in image_flies:
            score = photo_quality_organizer.calculate_score(image_path)
            print(f'{image_path}: {score}')
    
    # Get photos
    elif args.command == 'get_photos' :
        if args.sort_order and args.sort_order == 'ASC':
            is_desc = False
        else:
            is_desc = True           
        photos = photo_quality_organizer.get_photos_in_path(args.input_path, 20, is_desc)
        photo_paths = [item[0] for item in photos]

        # Discard similar photos, keep only the best ones
        similar_groups, remaining_images = find_similar_groups(photo_paths, 0.9)
        unique_image_paths = remaining_images
        for group in similar_groups:
            group.sort(key=photo_quality_organizer.calculate_score, reverse=True)
            unique_image_paths.append(group[0])
        for image_path in unique_image_paths:
            print(f'{image_path}: {photo_quality_organizer.calculate_score(image_path)}')
            scale_image_to_folder(image_path, args.output_path)

    # An exploratory attempt; you can open photo_organizer.db to view its data structure and write more interesting query logic
    elif args.command == 'diy_sql':
        sql = '''SELECT pp.file_path
                FROM photo_path pp
                JOIN all_categories ac ON pp.md5 = ac.md5
                JOIN all_scores ascore ON pp.md5 = ascore.md5
                WHERE ac.category = ?
                ORDER BY ascore.score DESC
                LIMIT ?'''
        params = ('Architecture', 100)
        photos = photo_quality_organizer.diy_sql(sql, params)
        photo_paths = [item[0] for item in photos]
        similar_groups, remaining_images = find_similar_groups(photo_paths, 0.6)
        unique_image_paths = remaining_images
        for group in similar_groups:
            group.sort(key=photo_quality_organizer.calculate_score, reverse=True)
            unique_image_paths.append(group[0])
        for image_path in unique_image_paths:
            print(f'{image_path}: {photo_quality_organizer.calculate_score(image_path)}')
            scale_image_to_folder(image_path, args.output_path)
        

    del photo_quality_organizer
    photo_quality_organizer = None




    
