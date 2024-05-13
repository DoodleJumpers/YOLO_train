from ultralytics import YOLO
from PIL import Image
import cv2
import sys


def main(img_name):
    model = YOLO("./fire_yolo.pt")
     
    result = model.predict(source=img_name, save=True, project="predicts",name=img_name[:-4]+"_predict.jpg")


if __name__ == "__main__":
    main(sys.argv[1])

