# This Python script is for training an image classification model.
import os
import platform
import random
import piexif
from PIL import Image


def get_modify_time(file_path):
    if platform.system() == 'Windows':
        # For Windows system, get the last modified time
        modify_time = os.path.getmtime(file_path)
    else:
        # For Unix/Linux systems, directly obtain the last modified time from the os.stat structure
        stat = os.stat(file_path)
        modify_time = stat.st_mtime

    # Convert to the number of seconds since January 1, 1970
    return int(modify_time)


# Generate a unique path in the target folder
def generate_unique_path(destination_folder, file_path):
    base_name, ext = os.path.splitext(os.path.basename(file_path))
    destination_path = os.path.join(destination_folder, f"{base_name}{ext}")

    while os.path.exists(destination_path):
        # If the path already exists, add a random number to create a unique path
        random_number = random.randint(0, 9999)
        destination_path = os.path.join(destination_folder, f"{base_name}_{random_number}{ext}")
    return destination_path


# Scale the image and save it to a folder
def scale_and_save_image(image_path, new_image_path , user_comment  = None):
    try:
        image = Image.open(image_path)
        
        # Try to extract EXIF data using piexif
        try:
            exif_dict = piexif.load(image.info['exif'])
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "Interop": {}, "thumbnail": None}

        if user_comment:
            try:                
                utf8_comment = user_comment.encode('utf-8')
                exif_dict['Exif'][piexif.ExifIFD.UserComment] = utf8_comment  
                exif_dict["0th"][40092] =  bytes(str(user_comment), encoding='utf-16')
            except Exception as e:
                print('error: failed to add user comment!', e)
        

        try:# Check and apply orientation information
            orientation = piexif.ImageIFD.Orientation
            if orientation in exif_dict['0th']:
                if exif_dict['0th'][orientation] == 3:
                    image = image.rotate(180, expand=True)
                    exif_dict['0th'][orientation] = 1  # Update orientation tag
                elif exif_dict['0th'][orientation] == 6:
                    image = image.rotate(270, expand=True)
                    exif_dict['0th'][orientation] = 1  # Update orientation tag
                elif exif_dict['0th'][orientation] == 8:
                    image = image.rotate(90, expand=True)
                    exif_dict['0th'][orientation] = 1  # Update orientation tag
        except (AttributeError, KeyError, IndexError):
            pass

         # See bug https://github.com/hMatoba/Piexif/issues/95
        try:
            del exif_dict['Exif'][piexif.ExifIFD.SceneType]
        except:
            pass

        try:
            exif_bytes = piexif.dump(exif_dict)
        except Exception as e:
            print('error: failed to dump exif data!', e)
            exif_bytes = None
        
        # Scale the image and save it to a folder
        width, height = image.size
        max_size = 1024
        scale = min(1.0, max_size / max(width, height))
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
        else:
            scaled_image = image.copy()

        # Save the image, while trying to preserve EXIF data
        if exif_bytes:
            scaled_image.save(new_image_path, exif=exif_bytes)
        else:
            scaled_image.save(new_image_path)
    except Exception as e:
        print(e)


# Correct image orientation
def correct_image_orientation(img):
 # Try to get EXIF data
    try:
        exif_dict = piexif.load(img.info['exif'])
        orientation = exif_dict['0th'][piexif.ImageIFD.Orientation]
    except (KeyError, TypeError):
        orientation = 1

    # Rotate the image based on EXIF orientation
    if orientation == 2:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        img = img.rotate(180)
    elif orientation == 4:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 6:
        img = img.rotate(-90, expand=True)
    elif orientation == 7:
        img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 8:
        img = img.rotate(90, expand=True)
    return img

# Scale the image and save it to a folder
def scale_image_to_folder(image_path, destination_folder):
    # Generate a unique target file path and save the image
    destination_file_path = generate_unique_path(destination_folder, image_path)
    scale_and_save_image(image_path, destination_file_path)

# Get all the images in the folder
def get_images(data_directory, include_subfolders=False):
    image_list = []
    # Traverse the folder to find image files
    for root, dirs, files in os.walk(data_directory):
        if 'sandbox' in dirs:
            dirs.remove('sandbox')
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith('.jpg') or file_lower.endswith('.jpeg'):
                image_path = os.path.join(root, file)
                image_list.append(image_path)        

        # If it does not include subfolders, only traverse the top-level directory
        if not include_subfolders:
            break
    return image_list







if __name__ == '__main__':
    image_path = '1.jpg'
    scale_and_save_image(image_path, 'test2.jpg', f'source:{image_path}')
    exif_dict = piexif.load('test2.jpg')  
    print(exif_dict)  
