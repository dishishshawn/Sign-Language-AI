import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Function to predict using YOLO and save result image
def predict_and_save_image(model, image_path):
    # Predict using YOLO
    results = model.predict([image_path], stream=True)
    result = next(results)  # Assuming there's only one result
    
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    
    # Save the result image
    result.save(filename="result.jpg")

# Function to handle image selection and prediction
def handle_image_upload():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Predict and save result image
        predict_and_save_image(model, file_path)
        
        # Update UI to display uploaded image and result image
        display_image(file_path)
        display_result_image("result.jpg")

# Function to display the uploaded image in the Tkinter window
def display_image(file_path):
    img = Image.open(file_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

# Function to display the result image in the Tkinter window
def display_result_image(result_image_path):
    result_img = Image.open(result_image_path)
    result_img.thumbnail((400, 400))
    result_img = ImageTk.PhotoImage(result_img)
    result_panel.config(image=result_img)
    result_panel.image = result_img

# Load your YOLO model
model = YOLO('./runs/classify/train13/weights/last.pt')

# Create the main window
root = tk.Tk()
root.title("Image Predictor")

# Set window dimensions
root.geometry("800x600")  # width x height

# Create and place components
upload_btn = tk.Button(root, text="Upload Image", command=handle_image_upload)
upload_btn.pack(pady=20)

# Panel to display uploaded image
panel = tk.Label(root)
panel.pack(side="left", padx=20)

# Panel to display result image
result_panel = tk.Label(root)
result_panel.pack(side="right", padx=20)

# Run the application
root.mainloop()
