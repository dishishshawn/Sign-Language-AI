from ultralytics import YOLO

model = YOLO('./runs/classify/train13/weights/last.pt') # load model

#results = model('./predict/testpic1.jpg') # precict from an image

results = model.predict(['./predict/testpic1.jpg'], stream=True)

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk