Automated Detection of Cardiac Arrhythmia using Recurrent Neural Network (LSTM & CNN)
This project implements a desktop application for the automated detection and classification of cardiac arrhythmias using deep learning models (Long Short-Term Memory - LSTM and Convolutional Neural Network - CNN). The application provides a user-friendly graphical interface (GUI) built with Tkinter to facilitate dataset management, model training, and performance evaluation.

Problem Statement
Cardiac arrhythmias, or irregular heartbeats, can range from benign to life-threatening conditions. Accurate and timely diagnosis is crucial for effective patient management. Manual interpretation of Electrocardiogram (ECG) signals by cardiologists is time-consuming and can be prone to human error. This project aims to automate this detection process using advanced deep learning techniques to provide a fast and reliable diagnostic aid.

Solution Overview
The application utilizes a deep learning pipeline consisting of data preprocessing, model training (LSTM and CNN), and comprehensive performance evaluation. It leverages the power of recurrent and convolutional neural networks to learn intricate patterns from ECG data and classify various cardiac conditions.

Features
Dataset Upload: Load ECG datasets (CSV format) into the application.

Data Preprocessing:

Handles missing values.

Encodes categorical labels.

Normalizes feature data.

Applies Principal Component Analysis (PCA) for dimensionality reduction.

Splits data into training and testing sets.

LSTM Model Training & Evaluation: Builds, trains, and evaluates a Long Short-Term Memory (LSTM) neural network.

CNN Model Training & Evaluation: Builds, trains, and evaluates a Convolutional Neural Network (CNN) model.

Performance Metrics: Calculates and displays Accuracy, Precision, Recall, F1-Score, Sensitivity, and Specificity for both models.

Confusion Matrices: Visualizes classification performance with seaborn heatmaps.

Training Graphs: Plots training accuracy and loss curves for both LSTM and CNN.

Performance Table: Generates an HTML table summarizing the performance of both models, which opens in a web browser.

Model Persistence: Saves trained models and their weights, allowing for faster re-evaluation without retraining.

Technologies Used:
Python 3.10

Tkinter: For the Graphical User Interface (GUI).

Pandas: For data manipulation and CSV handling.

NumPy: For numerical operations and array manipulation.

Scikit-learn (sklearn): For data preprocessing (LabelEncoder, normalize, PCA, train_test_split) and performance metrics.

Keras: High-level neural networks API.

TensorFlow: Backend for Keras.

Matplotlib: For plotting various graphs.

Seaborn: For enhanced data visualization, especially confusion matrices.

Git: Version control.

Git Large File Storage (Git LFS): For managing large dataset files within the repository.

Dataset
This project is designed to work with the MIT-BIH Arrhythmia Database, a widely recognized dataset for cardiac arrhythmia research. For ease of use with this application, it is recommended to use a preprocessed version of this dataset available in CSV format.

A suitable preprocessed dataset can be downloaded from Kaggle:

ECG Heartbeat Categorization Dataset: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Dataset Preparation Steps:

Download: Download the heartbeat.zip file from the Kaggle link above.

Extract: Unzip the downloaded file. You will find mitbih_train.csv and mitbih_test.csv.

Combine (Recommended): The application expects a single CSV file. You can combine mitbih_train.csv and mitbih_test.csv into a single file (e.g., combined_mitbih.csv) using a simple Python script (like combine_data.py provided in this repository):

import pandas as pd

train_df = pd.read_csv('mitbih_train.csv', header=None)
test_df = pd.read_csv('mitbih_test.csv', header=None)
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df.to_csv('combined_mitbih.csv', index=False, header=False)

Create Dataset Folder: In the root directory of this project, create a new folder named Dataset.

Place CSV: Place your combined_mitbih.csv (or just mitbih_train.csv if you prefer to test with a smaller subset) inside the Dataset folder.

Note on Dataset Columns: The Kaggle dataset files typically have 188 columns, with the last column (index 187) representing the heartbeat class. The application's preprocessing step (preprocessDataset) will automatically rename this last column to 279 for internal consistency. The classes are typically mapped as: 0: N (Normal), 1: S (Supraventricular Ectopic), 2: V (Ventricular Ectopic), 3: F (Fusion), 4: Q (Unclassified).

Setup and Installation
Follow these steps to get the project running on your local machine:

1. Clone the Repository
git clone https://github.com/ManideepK007/arrhythmia-detection.git
cd arrhythmia-detection

2. Install Git Large File Storage (Git LFS)
Since the dataset files are large, Git LFS is used to manage them.

Download & Install Git LFS: Visit https://git-lfs.github.com/ and follow the installation instructions for your operating system.

Initialize Git LFS in your repository:

git lfs install

Track .csv files with Git LFS:

git lfs track "*.csv"
git add .gitattributes
git commit -m "Configure Git LFS tracking for CSV files"

If you cloned the repo after these LFS configurations were pushed, git lfs pull should download the actual files.

3. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

4. Install Dependencies
With your virtual environment activated, install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib seaborn keras tensorflow

5. Prepare the Dataset
Follow the "Dataset Preparation Steps" outlined above to place your combined_mitbih.csv file into the Dataset/ folder.

Usage
Run the application:

python main.py

Upload Arrhythmia Dataset: Click this button to browse and select your combined_mitbih.csv file from the Dataset/ folder. The application will display the head of the dataset and an initial distribution plot.

Preprocess Dataset: Click this button to perform data cleaning, encoding, normalization, PCA, and train-test splitting. Details of the preprocessing will be displayed in the text area.

Run LSTM Algorithm: Click to train and evaluate the LSTM model. Training progress and metrics will appear in the text area, along with a confusion matrix plot.

Run CNN Algorithm: Click to train and evaluate the CNN model. Similarly, progress, metrics, and a confusion matrix will be shown.

LSTM & CNN Training Graph: After running both LSTM and CNN, click this button to view a comparative plot of their training accuracy and loss over epochs.

Performance Table: Click to generate an HTML table summarizing the performance metrics of both models. This table will open in your default web browser.

Exit: Close the application.

Expected Results
Upon successful execution, you will see:

Text output in the application's console area detailing dataset information, preprocessing steps, and model metrics.

Pop-up windows displaying bar charts for dataset distribution and confusion matrices for each model.

A comparative training graph showing accuracy and loss trends for both LSTM and CNN.

An HTML page in your web browser with a clear table comparing the performance metrics.

Future Enhancements
Real-time Prediction: Implement functionality to load a single ECG signal and predict its arrhythmia type.

Web Interface: Convert the Tkinter GUI to a web-based application (e.g., using Flask or Django for backend, and React/HTML/CSS/JS for frontend) for broader accessibility.

Hyperparameter Tuning: Implement automated hyperparameter optimization techniques to find the best model configurations.

Advanced Architectures: Explore more complex deep learning models, such as hybrid CNN-LSTM or Transformer-based networks.

Data Augmentation: Implement techniques to artificially expand the dataset to improve model robustness.

Dockerization: Containerize the application for easier deployment and environment consistency.

Developed by: ManideepK007 Contact: saimanideepk007@gmail.com
