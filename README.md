Currency Note Recognition
This project implements a Convolutional Neural Network (CNN) to recognize and classify Indian currency notes using TensorFlow and Keras. The model is trained to identify different denominations of Indian Rupee notes (e.g., ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000) based on images.
Table of Contents

Project Overview
Dataset
Requirements
Installation
Usage
Model Architecture
Training
Contributing
License

Project Overview
The goal of this project is to develop a deep learning model capable of classifying Indian currency notes. The model uses a CNN architecture to extract features from images and classify them into one of seven denominations. Data augmentation is applied to improve model robustness, and the model is trained and evaluated using a dataset of currency note images.
Dataset
The dataset used in this project is structured as follows:

Training Set: Located in dataset/Train, containing images of currency notes organized into folders by denomination (e.g., 1Hundrednote, 2Hundrednote, etc.).
Test Set: Located in dataset/Test, with a similar structure for validation.
The dataset is provided in a zipped format (CNN_Dataset.zip) and is extracted during execution.

Note: The dataset is not included in this repository due to size constraints. You can obtain it separately and place CNN_Dataset.zip in the project root directory.
Requirements

Python 3.8+
TensorFlow 2.x
Keras
NumPy
Matplotlib
Jupyter Notebook (to run CNN.ipynb)

Installation

Clone the repository:
git clone https://github.com/your-username/currency-note-recognition.git
cd currency-note-recognition


Install the required Python packages:
pip install tensorflow numpy matplotlib


Place the dataset (CNN_Dataset.zip) in the project root directory.

Extract the dataset by running the first cell in CNN.ipynb or manually unzip it to create a dataset folder.


Usage

Open the Jupyter Notebook:
jupyter notebook CNN.ipynb


Run the notebook cells sequentially to:

Extract the dataset.
Preprocess images using data augmentation.
Build and compile the CNN model.
Train the model on the training dataset.
Evaluate the model on the test dataset.


To use the trained model for inference:

Modify the notebook to include a prediction function for new images.
Example code for inference can be added to the notebook (not currently included).



Model Architecture
The CNN model consists of:

Three convolutional layers with ReLU activation (32, 64, and 128 filters, respectively).
MaxPooling layers after each convolutional layer.
A flatten layer to transition to dense layers.
A dense layer with 128 units and ReLU activation.
A dropout layer (0.5) to prevent overfitting.
A final dense layer with softmax activation for classification (7 classes).

The model is compiled with:

Optimizer: Adam (learning rate = 0.001)
Loss: Categorical Crossentropy
Metric: Accuracy

Training

The model is trained for 27 epochs with a batch size of 32.
Data augmentation (rotation, zoom, horizontal flip) is applied to the training set.
The training and validation accuracy/loss are plotted during training (requires additional plotting code).

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
