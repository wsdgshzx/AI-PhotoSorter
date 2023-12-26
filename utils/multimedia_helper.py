import piexif
from datetime import datetime
import re

from database_manager import DatabaseManager
from utils.comm_util import get_modify_time
from utils.file_md5 import *

def get_photo_exif(image_path):
    return MultimediaHelper.get_instance().get_photo_exif(image_path)

def get_photo_create_time(image_path):
    return MultimediaHelper.get_instance().get_photo_create_time(image_path)    

class MultimediaHelper:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.db_manager = DatabaseManager.get_instance()        

    def get_photo_exif(self, photo_path): 
        md5 = get_image_md5(photo_path)
        need_save = False

        photo_exif = self.db_manager.load_photo_info(md5)        
        if photo_exif is None:                   
            photo_exif = self._get_photo_exif(photo_path)
            if photo_exif is None:
                photo_exif = {}
            need_save = True 
        original_time = photo_exif.get('date_time_original')
        if original_time is None or original_time <= 0:
            photo_exif['date_time_original'] = get_modify_time(photo_path)   
            need_save = True 
        if need_save:             
            self.db_manager.save_photo_info(md5, photo_exif)
        return photo_exif  

    def get_photo_create_time(self, photo_path):
        md5 = get_image_md5(photo_path)
        photo_create_time = self.db_manager.load_photo_created_time(md5)
        if photo_create_time is None:
            exif = self.get_photo_exif(photo_path)
            photo_create_time = exif.get('date_time_original')        
        assert photo_create_time is not None, 'photo_create_time is None'    
        return photo_create_time

    @staticmethod
    def _get_photo_exif(photo_path):             
        def convert_rational_to_decimal(value):
            # Check if the value is a tuple or list containing two elements
            if not (value and isinstance(value, (tuple, list)) and len(value) == 2):
                return None

            # Check if the numerator and denominator are numerical
            numerator, denominator = value
            if not all(isinstance(x, (int, float)) for x in [numerator, denominator]):
                return None

            # Avoid division by zero
            if denominator == 0:
                return None

            return numerator / denominator

        def convert_gps_to_decimal(gps_data):
            if not gps_data or len(gps_data) < 2:
                return None
            degrees = convert_rational_to_decimal(gps_data[0])
            minutes = convert_rational_to_decimal(gps_data[1])
            seconds = convert_rational_to_decimal(gps_data[2]) if len(gps_data) > 2 else 0
            if degrees is None or minutes is None:
                return None
            return degrees + (minutes / 60) + (seconds / 3600)  
        

        try:
            exif_data = piexif.load(photo_path)
            has_exif = any(exif_data[ifd] for ifd in exif_data)
            if not has_exif: # If there is no Exif information, return None
                return None
            
            photo_info = {}
            # Extract camera manufacturer and model
            try:
                make_s = exif_data['0th'].get(piexif.ImageIFD.Make, b'').decode('utf-8', 'ignore')
                photo_info['make'] = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', make_s).strip()
                model_s = exif_data['0th'].get(piexif.ImageIFD.Model, b'').decode('utf-8', 'ignore')
                photo_info['model'] = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', model_s).strip()
            except Exception as e:
                print(f"Error getting camera info: {e}")

            # Extract exposure time, aperture value, etc.
            try:
                photo_info['exposure_time'] = str(convert_rational_to_decimal(exif_data['Exif'].get(piexif.ExifIFD.ExposureTime)))
                photo_info['f_number'] = str(convert_rational_to_decimal(exif_data['Exif'].get(piexif.ExifIFD.FNumber)))
                photo_info['iso'] = exif_data['Exif'].get(piexif.ExifIFD.ISOSpeedRatings)
            except Exception as e:
                print(f"Error getting exposure info: {e}")

            # Extract shooting time
            try:
                date_time_original = exif_data['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
                if date_time_original:
                    photo_info['date_time_original'] = int(datetime.strptime(date_time_original.decode('utf-8'), '%Y:%m:%d %H:%M:%S').timestamp())
            except Exception as e:
                print(f"Error getting original date time: {e}")

            # Extract GPS information
            try:
                gps_latitude = convert_gps_to_decimal(exif_data['GPS'].get(piexif.GPSIFD.GPSLatitude))
                gps_longitude = convert_gps_to_decimal(exif_data['GPS'].get(piexif.GPSIFD.GPSLongitude))
                gps_altitude = convert_rational_to_decimal(exif_data['GPS'].get(piexif.GPSIFD.GPSAltitude))
                if gps_latitude and gps_longitude and gps_altitude:
                    photo_info['gps_latitude'] = gps_latitude
                    photo_info['gps_longitude'] = gps_longitude
                    photo_info['gps_altitude'] = gps_altitude
            except Exception as e:
                print(f"Error getting GPS info: {e}")

            # Extract lens manufacturer and model
            try:
                photo_info['lens_make'] = exif_data['Exif'].get(piexif.ExifIFD.LensMake, b'').decode('utf-8', 'ignore')
                photo_info['lens_model'] = exif_data['Exif'].get(piexif.ExifIFD.LensModel, b'').decode('utf-8', 'ignore')
            except Exception as e:
                print(f"Error getting lens info: {e}")

            # Extract focal length, shutter speed, etc.
            try:
                photo_info['focal_length'] = str(convert_rational_to_decimal(exif_data['Exif'].get(piexif.ExifIFD.FocalLength, (0, 1))))
                photo_info['shutter_speed_value'] = str(convert_rational_to_decimal(exif_data['Exif'].get(piexif.ExifIFD.ShutterSpeedValue, (0, 1))))
                photo_info['aperture_value'] = str(convert_rational_to_decimal(exif_data['Exif'].get(piexif.ExifIFD.ApertureValue, (0, 1))))
                photo_info['brightness_value'] = str(convert_rational_to_decimal(exif_data['Exif'].get(piexif.ExifIFD.BrightnessValue, (0, 1))))
            except Exception as e:
                print(f"Error getting advanced exposure info: {e}")

            # Extract flash status, white balance, etc.
            try:
                photo_info['flash'] = exif_data['Exif'].get(piexif.ExifIFD.Flash, 0)
                photo_info['white_balance'] = exif_data['Exif'].get(piexif.ExifIFD.WhiteBalance, 0)
                photo_info['metering_mode'] = exif_data['Exif'].get(piexif.ExifIFD.MeteringMode, 0)
                photo_info['exposure_mode'] = exif_data['Exif'].get(piexif.ExifIFD.ExposureMode, 0)
                photo_info['exposure_program'] = exif_data['Exif'].get(piexif.ExifIFD.ExposureProgram, 0)
            except Exception as e:
                print(f"Error getting camera settings: {e}")

            return photo_info
        except Exception as e:
            print(f"Error processing file {photo_path}: {e}")
            return None
        
    @staticmethod
    def update_exif_day(photo_path, date_per_day):
        try:
            exif_data = piexif.load(photo_path)
            has_update = False

            if not exif_data:
                return

            # Ensure the date format is YYYY:MM:DD
            if not datetime.strptime(date_per_day, "%Y:%m:%d"):
                raise ValueError("Date format must be YYYY:mm:dd")

            # Determine if there is Exif information
            if exif_data and 'Exif' in exif_data:
                # Read the original shooting time
                date_time_original = exif_data['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
                if date_time_original:
                    # Change the date, but keep the original hours, minutes, and seconds
                    date_time_original = date_time_original.decode('utf-8')
                    new_date_time_original = f"{date_per_day} {date_time_original[11:]}"
                    exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal] = new_date_time_original.encode('utf-8')
                    exif_bytes = piexif.dump(exif_data)                    
                    piexif.insert(exif_bytes, photo_path)
                    has_update = True

            # If Exif information or the DateTimeOriginal field does not exist, create or update the field
            if not has_update:
                if 'Exif' not in exif_data:
                    exif_data['Exif'] = {}
                # Use the default time "00:00:00" to complete the time part
                exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal] = f"{date_per_day} 00:00:00".encode('utf-8')
                exif_bytes = piexif.dump(exif_data)                    
                piexif.insert(exif_bytes, photo_path)

        except Exception as e:
            print(f"Error updating date time original: {e}")

    @staticmethod
    def update_exif_date_time(photo_path, date_time_original):
        try:
            exif_data = piexif.load(photo_path)
            if not exif_data:
                return
                
            # Ensure the date format is YYYY:MM:DD
            if not datetime.strptime(date_time_original, "%Y:%m:%d %H:%M:%S"):
                raise ValueError("Date format must be %Y:%m:%d %H:%M:%S")

            # If Exif information or the DateTimeOriginal field does not exist, create or update the field
            if 'Exif' not in exif_data:
                exif_data['Exif'] = {}
            # Use the default time "00:00:00" to complete the time part
            exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal] = date_time_original.encode('utf-8')
            exif_bytes = piexif.dump(exif_data)                    
            piexif.insert(exif_bytes, photo_path)

        except Exception as e:
            print(f"Error updating date time original: {e}")


if __name__ == "__main__":
    get_photo_create_time('demo.jpg') 
