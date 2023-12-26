import argparse
import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from torchvision.transforms.functional import InterpolationMode
from database_manager import DatabaseManager

from logger_unit import CustomLogger
from utils.comm_util import *
from utils.file_md5 import get_image_md5

# class RandomRotation90:
#     def __call__(self, x):
#         angle = random.choice([0, 90, 180, 270])
#         return TF.rotate(x, angle)
    
class RandomResizeCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.5, 1.0), ratio=(3./4., 4./3.), interpolation=InterpolationMode.BICUBIC):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)

    def __call__(self, img):
        # Randomly choose a size within the 256 to 512 pixel range
        target_size = random.randint(256, 512)
        # Scale the image to that size
        img = transforms.Resize(target_size, interpolation=self.interpolation)(img)
        # Then crop the image
        return super().__call__(img)

class ImageClassifier:
    def __init__(self, batch_size=32, validation_split=0.2, learning_rate=0.0005, epochs = 10, data_dir=None, model_filename='class_model.pth'):

        self.logger = CustomLogger('ImageClassifier')
        self.db_manager = DatabaseManager.get_instance()   
        
        self.model_file_path = os.path.join(os.getcwd(), 'train_data', model_filename)  # Path where the model is saved
        self.model_class_file_path = os.path.join(os.getcwd(), 'train_data', model_filename+'.class')  # Path where the model is saved
        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), 'train_data', 'class')
        self.data_dir = data_dir

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.class_names = None        
        self.db_category_cache = {}

    def load_data(self):
        if not os.path.exists(self.model_file_path):
            self.logger.info('Start to train model')
            num_classes = self.count_subdirectories(self.data_dir)
            self._load_data()           
            self.initialize_model(num_classes) # Initialize the model            
            self.train_model(self.epochs, self.learning_rate)# Begin the training process           
            self.save_model() # Save the model
        else:
            self.load_model()   



    def _load_data(self):
        # Transformations for the training data
        train_transform = transforms.Compose([
            # transforms.Resize(256),  # Adjust the shorter side to 256 pixels
            RandomResizeCrop(224),  # Randomly scale and then crop to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transformations for the validation data (without random transformations)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Loading dataset with different transformations for training and validation
        full_dataset = ImageFolder(self.data_dir) #transform=transforms.Resize((512, 512)
        self.class_names = full_dataset.classes  # Get the class names
        train_size = int((1 - self.validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply transformations
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        # Creating data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

    def initialize_model(self, num_classes):
        self.model = models.resnet50(pretrained=True)
        # Replace the last fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)

    def train_model(self, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            running_loss = 0.0
            self.model.train() 
            for i, (inputs, labels) in enumerate(self.train_loader, 0):                
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}")
            self.validate_model(criterion)

    def validate_model(self, criterion):
        self.model.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file_path)
        with open(self.model_class_file_path, 'w') as f:
            for class_name in self.class_names:
                f.write(class_name + '\n')

    def load_model(self):
        if self.class_names is None and os.path.exists(self.model_class_file_path):
            with open(self.model_class_file_path, 'r') as f:
                self.class_names = f.read().splitlines()
        self.model = models.resnet50(pretrained=False).to(self.device)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)
        


    def predict_category(self, image_path):
        if self.model is None:
            print('Please load model first')
            return None        
        # Load and preprocess the image
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)

        # Make a prediction
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        
        idx = predicted.item()
        return self.class_names[idx]
    
    def detect_image_category(self, image_path):
        if image_path is None:
            return None
        md5 = get_image_md5(image_path)
        if md5 in self.db_category_cache:
            return self.db_category_cache[md5] 
              
        category = self.db_manager.load_category(md5)
        if category is None: 
            try:          
                category = self.predict_category(image_path)
                self.db_category_cache[md5] = category
            except Exception as e:
                self.logger.error(f"Predict category error: {e}")
                category = ''           

            if len(self.db_category_cache) > 100:
                self.db_manager.batch_save_category(self.db_category_cache)
                self.db_category_cache.clear()

        return category
        
    
    @staticmethod
    def count_subdirectories(directory):
        subdir_count = 0
        for entry in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, entry)):
                subdir_count += 1
        return subdir_count



##############################################################################################################





if __name__ == '__main__':
    batch_size = 32
    validation_split = 0.2
    learning_rate = 0.00005
    epochs = 5


    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, help='Command to execute')
    parser.add_argument('--input_path', type=str, default='E:\\test\\', help='Path to input file')
    parser.add_argument('--output_path', type=str, default='E:\\output\\', help='Path to output file')    
    args,_ = parser.parse_known_args()  
    print(args)


    # Creating an instance of ImageClassifier with specified parameters.
    # 'batch_size' controls the number of images processed at once during training.
    # 'validation_split' determines the proportion of data used for validation.
    # 'learning_rate' and 'epochs' are hyperparameters for the training process.
    classifier = ImageClassifier(batch_size=batch_size, validation_split=validation_split, learning_rate=learning_rate, epochs=epochs)

    # Loading the data into the classifier. This typically involves reading images, preprocessing them, 
    # and organizing them into training and validation sets.
    classifier.load_data()

    # Retrieving a list of image paths from the specified input directory.
    image_list = get_images(args.input_path, True)

    # Iterating over each image in the list.
    for image_path in image_list:
        # Detecting the category of the current image using the classifier.
        # feeding it into the trained model, and interpreting the output.
        class_name = classifier.detect_image_category(image_path)

        print(f"image_path: {image_path}, class_name: {class_name}")
