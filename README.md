# Sign Language Prediction App

This application leverages the YOLOv8 deep learning model to identify and predict sign language gestures. It features a user-friendly GUI for selecting images and visualizing the prediction results. The application includes three main components:

1. **Model Training** (`main.py`)  
2. **Prediction Script** (`predict.py`)  
3. **User Interface** (`UI.py`)  

## Project Structure

- **`datasets/`**: Contains datasets for training and testing. (Not in repo because they are too large, but can be found at https://github.com/grassknoted/Unvoiced/tree/master/Test%20Images)
- **`main.py`**: Script for training the YOLOv8 model with custom sign language datasets.
- **`predict.py`**: Performs predictions on images and saves the output.
- **`UI.py`**: Provides a graphical user interface to upload images, predict, and display results.
- **`runs/`**: Stores trained model weights and logs.

---

## How to Use

### Prerequisites

- Python 3.8+
- Required Python libraries (install via `pip`):  
  ```bash
  pip install ultralytics tkinter pillow numpy
  ```

### Steps to Run

1. **Train the Model (Optional)**  
   If you want to train the model with custom data:
   ```bash
   python main.py
   ```
   Ensure your dataset is prepared and configured in `main.py`.

2. **Make Predictions**  
   Run the prediction script:
   ```bash
   python predict.py
   ```
   This will load a sample image (`testpic1.jpg`) from the `predict/` folder and save the result as `result.jpg`.

3. **Use the GUI**  
   Start the graphical interface for image prediction:
   ```bash
   python UI.py
   ```
   - Click **"Upload Image"** to select an image.
   - The app will display both the input and result images side by side.

---

## Areas for Improvement

1. **Model Performance**  
   - Experiment with hyperparameters (e.g., epochs, batch size, learning rate).
   - Use a larger and more diverse dataset to improve accuracy.

2. **GUI Enhancements**  
   - Add real-time webcam support for live predictions.
   - Improve UI layout and provide detailed feedback about predictions.

3. **Model Integration**  
   - Include support for batch predictions.
   - Implement multiple model versions and allow users to choose models.

4. **Code Modularization**  
   - Abstract common functionalities (e.g., prediction logic) into reusable modules.
   - Add logging for better debugging and monitoring.

5. **Documentation**  
   - Provide detailed examples and dataset preparation guidelines.
   - Include pre-trained models for easy testing.

---

## License

This project is released under the MIT License.
