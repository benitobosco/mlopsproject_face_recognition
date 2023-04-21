import os
import sys
from src.exception import CustomException
from src.logger import logging
import cv2 
from matplotlib import pyplot as plt
from dataclasses import dataclass
import shutil


face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')

def get_cropped_image_if_1_face_detected(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if len(faces) == 1:
            return roi_gray

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts',"dataset")
    cropped_data_path: str=os.path.join('artifacts',"cropped")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def cropped_images(self, img_dirs):
        cropped_image_dirs = []
        celebrity_file_names_dict = {}
        for img_dir in img_dirs:
            count = 1
            celebrity_name = img_dir.split('\\')[-1]
            celebrity_file_names_dict[celebrity_name] = []
            for entry in os.scandir(img_dir):
                roi_gray = get_cropped_image_if_1_face_detected(entry.path)
                if roi_gray is not None:
                    cropped_folder = self.ingestion_config.cropped_data_path + "/" + celebrity_name
                    if not os.path.exists(cropped_folder):
                        os.makedirs(cropped_folder)
                        cropped_image_dirs.append(cropped_folder)
                        print("Generating cropped images in folder: ",cropped_folder)
                    cropped_file_name = celebrity_name + str(count) + ".png"
                    cropped_file_path = cropped_folder + "/" + cropped_file_name
                    cv2.imwrite(cropped_file_path, roi_gray)
                    celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                    count += 1


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            source_dir = "notebook\dataset"
            destination_dir = self.ingestion_config.raw_data_path
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

            logging.info('Read the dataset as dataframe')

            img_dirs = []
            for entry in os.scandir(self.ingestion_config.raw_data_path):
                if entry.is_dir():
                    img_dirs.append(entry.path)

            if os.path.exists(self.ingestion_config.cropped_data_path):
                shutil.rmtree(self.ingestion_config.cropped_data_path)

            os.mkdir(self.ingestion_config.cropped_data_path)

            logging.info("Cropped images initiated")

            self.cropped_images(img_dirs)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.raw_data_path,
                self.ingestion_config.cropped_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
