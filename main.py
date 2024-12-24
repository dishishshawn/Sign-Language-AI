import os
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt') #  load a pretrained model (recommended for training)

model.train(data='D:/API Project (Sign-Language)/YoloV8-Test/weather_dataset', epochs=20, imgsz=128) #train15 is 128x128
