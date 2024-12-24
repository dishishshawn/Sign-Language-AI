import os
import numpy as np
from ultralytics import YOLO

model = YOLO('D:/API Project (Sign-Language)/YoloV8-Test/runs/classify/train13/weights/last.pt') #  load a pretrained model (recommended for training)

model.train(data='D:/API Project (Sign-Language)/YoloV8-Test/improve_dataset', epochs=20, imgsz=64)
