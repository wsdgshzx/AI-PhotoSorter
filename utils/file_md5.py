import os
import hashlib
from database_manager import DatabaseManager

md5_fast_model = True

def get_image_md5(image_path):
    return ImageHashSingleton.get_instance().get_md5(image_path)

def get_image_path_by_md5(md5):
    return ImageHashSingleton.get_instance().get_file_path(md5)

class ImageHashSingleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.image_md5_cache = {} # Used for caching the image's hash
        self.db_manager = DatabaseManager.get_instance()
        self.md5_db_cache = {}

    def __del__(self):
        if len(self.md5_db_cache) > 0:
            self.db_manager.batch_save_md5(self.md5_db_cache)
            self.md5_db_cache = {}
    
    def get_md5(self, image_path):
        # Add caching
        image_path = os.path.normpath(image_path)
        if image_path in self.image_md5_cache:
            return self.image_md5_cache[image_path]
        # Read from the database       
        md5 = self.db_manager.load_md5_by_file_path(image_path)
        # If not in the database, then calculate md5
        if md5 is None:            
            md5 = self._calculate_md5(image_path)
            if md5_fast_model:
                self.md5_db_cache[image_path] = md5
            else:
                self.db_manager.save_md5(image_path, md5)
        self.image_md5_cache[image_path] = md5

        if md5_fast_model:
            if len(self.md5_db_cache) > 500:
                self.db_manager.batch_save_md5(self.md5_db_cache)
                self.md5_db_cache = {}

        return md5
    
    def get_file_path(self, md5):
        return self.db_manager.get_file_path(md5)

    # Calculate the file's md5
    @staticmethod
    def _calculate_md5(file_path):        
        with open(file_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        return md5