from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import webbrowser
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, MaxPooling2D, Activation, Flatten, Convolution2D
from keras.models import model_from_json
import pickle

# FIX: Changed import for to_categorical to reflect Keras 3.x and TensorFlow backend
from tensorflow.keras.utils import to_categorical 


# Main Tkinter window setup
main = tkinter.Tk()
main.title("Automated Detection of Cardiac Arrhythmia using Recurrent Neural Network")
main.geometry("1200x1200")

# Global variables for data and models
global X_train, X_test, y_train, y_test, pca
global model, dataset
global filename
global X, Y

# Lists to store performance metrics
accuracy = []
precision = []
recall = []
fscore = []
sensitivity = []
specificity = []

# Labels for different cardiac conditions (updated to reflect common MIT-BIH processed dataset categories)
# Ensure these match the actual labels in your dataset after LabelEncoding
labels = ['Normal (N)', 'Supraventricular Ectopic (S)', 'Ventricular Ectopic (V)', 'Fusion (F)', 'Unclassified (Q)']


def uploadDataset():
    """
    Function to upload a dataset (CSV file) for arrhythmia detection.
    It loads the dataset, displays its head, and plots the distribution of labels.
    """
    global filename, dataset
    text.delete('1.0', END)  # Clear the text widget
    filename = filedialog.askopenfilename(initialdir="Dataset") # Open file dialog
    
    if not filename: # Handle case where user cancels file selection
        text.insert(END, "Dataset upload cancelled.\n")
        pathlabel.config(text="No dataset loaded")
        return

    text.insert(END, str(filename) + " Dataset Loaded\n\n") # Display loaded file path
    pathlabel.config(text=str(filename) + " Dataset Loaded") # Update path label
    
    try:
        dataset = pd.read_csv(filename) # Read the CSV into a pandas DataFrame
        text.insert(END, str(dataset.head())) # Display the first few rows of the dataset

        # Plot the distribution of the target variable (column '279' after renaming in preprocess)
        # For this to work on initial upload, the '279' column must exist or you need to
        # know the actual label column index/name before preprocessing.
        # As per the Kaggle dataset, the label column is the last one, and is numeric.
        # So we can plot it directly, or wait until after preprocessing if needed.
        # For now, let's assume the last column is the one to plot initially if it exists.
        if dataset.shape[1] > 0: # Check if dataset is not empty
            # Assuming the last column is the target column
            temp_label_col = dataset.iloc[:, -1]
            if pd.api.types.is_numeric_dtype(temp_label_col):
                label_counts = temp_label_col.value_counts().sort_index()
                plt.figure(figsize=(8, 6))
                label_counts.plot(kind="bar")
                plt.title("Initial Distribution of Cardiac Conditions (Raw Labels)")
                plt.xlabel("Raw Label ID")
                plt.ylabel("Number of Samples")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            else:
                text.insert(END, "\nCannot plot initial distribution: Last column is not numeric.\n")
        
    except Exception as e:
        text.insert(END, f"Error loading dataset: {e}\n")
        pathlabel.config(text="Error loading dataset")

def preprocessDataset():
    """
    Function to preprocess the uploaded dataset.
    Steps include:
    1. Filling missing values with 0.
    2. Encoding categorical labels using LabelEncoder.
    3. Separating features (X) and target (Y).
    4. Shuffling the data.
    5. Normalizing features using min-max normalization.
    6. Applying PCA for dimensionality reduction to 40 components.
    7. One-hot encoding the target variable.
    8. Reshaping X for LSTM input.
    9. Splitting the data into training and testing sets (80/20 split).
    10. Displaying preprocessing details in the text widget.
    """
    global X, Y, dataset, pca
    global X_train, X_test, y_train, y_test
    global labels # Re-declare global if modifying in this function
    text.delete('1.0', END) # Clear the text widget

    if dataset is None:
        text.insert(END, "No dataset loaded. Please upload a dataset first.\n")
        return

    try:
        # Fill NaN values with 0
        dataset.fillna(0, inplace=True)

        # Assuming the Kaggle dataset's last column (index -1 or 187) is the label
        # Rename the last column to '279' for compatibility with existing code
        if dataset.columns[-1] != '279':
            dataset.rename(columns={dataset.columns[-1]: '279'}, inplace=True)
        
        le = LabelEncoder()
        # Encode the target variable (column '279')
        dataset["279"] = pd.Series(le.fit_transform(dataset["279"].astype(str)))

        # Update global labels to match the new encoded labels after transformation
        # This assumes the labels are sorted numerically after encoding.
        # This will be based on the unique values found in the '279' column after encoding.
        # If using the Shayan Fazeli dataset, the mapping is 'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4
        # We define them explicitly for clarity and matching.
        labels = ['Normal (N)', 'Supraventricular Ectopic (S)', 'Ventricular Ectopic (V)', 'Fusion (F)', 'Unclassified (Q)']


        # Separate features (X) and target (Y)
        temp = dataset.values
        X = temp[:, 0:temp.shape[1] - 1]
        Y = temp[:, temp.shape[1] - 1]

        # Shuffle the dataset
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        text.insert(END, "Original Y (after shuffling, pre-one-hot-encoding):\n" + str(np.unique(Y)) + "\n\n")

        # Normalize features (Min-Max scaling implicitly by sklearn.preprocessing.normalize)
        X = normalize(X)

        # Apply PCA for dimensionality reduction
        # Ensure n_components is less than or equal to the number of features after splitting X
        n_components_pca = min(40, X.shape[1]) # Adjust if X.shape[1] is less than 40
        pca = PCA(n_components = n_components_pca)
        X = pca.fit_transform(X)

        # One-hot encode the target variable
        Y = to_categorical(Y) # This will create Y with shape (num_samples, num_classes)

        # Reshape X for LSTM input (samples, timesteps, features)
        # Here, timesteps is X.shape[1] (after PCA), and features per timestep is 1
        XX = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2, random_state=42)

        # Display preprocessing summary
        text.insert(END, "Total records found in dataset : " + str(X.shape[0]) + "\n\n")
        text.insert(END, "Different diseases found in dataset (mapped labels):\n")
        text.insert(END, str(labels) + "\n\n")
        text.insert(END, "Dataset Train & Test Split Details\n\n")
        text.insert(END, "Total records used to train LSTM & CNN :" + str(X_train.shape[0]) + "\n")
        text.insert(END, "Total records used to test LSTM & CNN :" + str(X_test.shape[0]) + "\n")
        text.insert(END, f"X_train shape: {X_train.shape}\n")
        text.insert(END, f"y_train shape: {y_train.shape}\n")
        text.update_idletasks()

    except Exception as e:
        text.insert(END, f"Error during preprocessing: {e}\n")
        return

def calculateMetrics(algorithm, predict, y_test):
    """
    Calculates and displays various performance metrics for a given algorithm.
    Metrics include: Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity.
    Also plots the confusion matrix.

    Args:
        algorithm (str): Name of the algorithm (e.g., "LSTM", "CNN").
        predict (numpy.ndarray): Predicted labels.
        y_test (numpy.ndarray): True labels.
    """
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(END, algorithm + " Accuracy : " + str(round(a, 2)) + "%\n")
    text.insert(END, algorithm + " Precision : " + str(round(p, 2)) + "%\n")
    text.insert(END, algorithm + " Recall : " + str(round(r, 2)) + "%\n")
    text.insert(END, algorithm + " FScore : " + str(round(f, 2)) + "%\n")

    conf_matrix = confusion_matrix(y_test, predict)
    
    # Sensitivity (Recall for each class) and Specificity (True Negative Rate)
    # For multi-class, these are usually calculated per-class or as averages.
    # The original code's sensitivity/specificity calculation was for binary.
    # Let's provide a general idea or stick to the macro-averaged ones if that's the intent.
    # For simplicity, if we want a single 'sensitivity' and 'specificity' for multi-class,
    # it's often derived from macro averages or by considering one class as positive.
    # Here, let's keep the original logic for 2x2 matrix calculation, if applicable,
    # or consider the overall performance from precision/recall/f1.
    
    # A more robust multi-class approach for sensitivity/specificity would iterate through classes:
    sensitivities = []
    specificities = []
    for i in range(conf_matrix.shape[0]):
        tp_i = conf_matrix[i, i]
        fn_i = np.sum(conf_matrix[i, :]) - tp_i
        fp_i = np.sum(conf_matrix[:, i]) - tp_i
        tn_i = np.sum(conf_matrix) - tp_i - fn_i - fp_i

        sens = tp_i / (tp_i + fn_i) if (tp_i + fn_i) != 0 else 0
        spec = tn_i / (tn_i + fp_i) if (tn_i + fp_i) != 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)
    
    avg_sensitivity = np.mean(sensitivities) if sensitivities else 0
    avg_specificity = np.mean(specificities) if specificities else 0

    text.insert(END, algorithm + ' Average Sensitivity : ' + str(round(avg_sensitivity, 2)) + "\n")
    text.insert(END, algorithm + ' Average Specificity : ' + str(round(avg_specificity, 2)) + "\n\n")
    sensitivity.append(avg_sensitivity)
    specificity.append(avg_specificity)
    text.update_idletasks()

    # Plot confusion matrix
    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g", cbar_kws={'label': 'Count'})
    ax.set_ylim([0, len(labels)])
    plt.title(algorithm + " Confusion Matrix", fontsize=16)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.show()

def runLSTM():
    """
    Function to build, train, and evaluate an LSTM model.
    If a pre-trained model exists, it loads it; otherwise, it trains a new one.
    It then predicts on the test set and calls calculateMetrics.
    """
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, sensitivity, specificity
    text.delete('1.0', END)

    # Ensure dataset is preprocessed before running model
    if 'X_train' not in globals() or X_train is None:
        text.insert(END, "Dataset not preprocessed. Please click 'Preprocess Dataset' first.\n")
        return

    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    sensitivity.clear()
    specificity.clear()

    lstm = None

    # Ensure 'model' directory exists
    if not os.path.exists('model'):
        os.makedirs('model')

    if os.path.exists('model/lstm_model.json'):
        text.insert(END, "Loading pre-trained LSTM model...\n")
        try:
            with open('model/lstm_model.json', "r") as json_file:
                loaded_model_json = json_file.read()
            lstm = model_from_json(loaded_model_json)
            lstm.load_weights("model/lstm_model_weights.h5")
            text.insert(END, "LSTM model loaded successfully.\n\n")
        except Exception as e:
            text.insert(END, f"Error loading LSTM model: {e}. Training a new model.\n\n")
            lstm = None
    
    if lstm is None:
        text.insert(END, "Training a new LSTM model...\n")
        lstm_model = Sequential()
        lstm_model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(100, activation='relu'))
        lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        text.insert(END, "Starting LSTM training (this may take a while)...\n")
        main.update_idletasks() # Update GUI to show message
        
        hist = lstm_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=0)
        
        lstm_model.save_weights('model/lstm_model_weights.h5')
        model_json = lstm_model.to_json()
        with open("model/lstm_model.json", "w") as json_file:
            json_file.write(model_json)
        
        with open('model/lstm_history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
        
        lstm = lstm_model
        text.insert(END, "LSTM model training complete and saved.\n\n")

    if lstm:
        text.insert(END, "LSTM Model Summary:\n")
        lstm.summary(print_fn=lambda x: text.insert(END, x + '\n'))
        
        text.insert(END, "\nPerforming LSTM prediction and metric calculation...\n")
        predict = lstm.predict(X_test)
        predict = np.argmax(predict, axis=1)
        testY = np.argmax(y_test, axis=1)
        calculateMetrics("LSTM", predict, testY)
    else:
        text.insert(END, "LSTM model could not be loaded or trained.\n")


def runCNN():
    """
    Function to build, train, and evaluate a CNN model.
    It reshapes X for CNN input, splits data, and handles model loading/training similar to LSTM.
    It then predicts on the test set and calls calculateMetrics.
    """
    global X, Y
    global accuracy, precision, recall, fscore, sensitivity, specificity
    
    if 'X' not in globals() or X is None or 'Y' not in globals() or Y is None:
        text.insert(END, "Dataset not preprocessed. Please click 'Preprocess Dataset' first.\n")
        return

    # Reshape X for CNN input (samples, height, width, channels)
    # The PCA output is X.shape[1] features, treat as 1D image of that length
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    
    # CNN needs its own train/test split because the input shape is different (4D for CNN vs 3D for LSTM)
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(XX, Y, test_size=0.2, random_state=42)

    cnn = None

    if not os.path.exists('model'):
        os.makedirs('model')

    if os.path.exists('model/cnn_model.json'):
        text.insert(END, "Loading pre-trained CNN model...\n")
        try:
            with open('model/cnn_model.json', "r") as json_file:
                loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
            cnn.load_weights("model/cnn_model_weights.h5")
            text.insert(END, "CNN model loaded successfully.\n\n")
        except Exception as e:
            text.insert(END, f"Error loading CNN model: {e}. Training a new model.\n\n")
            cnn = None
    
    if cnn is None:
        text.insert(END, "Training a new CNN model...\n")
        cnn_model = Sequential()
        # Convolution2D expects (rows, cols, channels) for input_shape. Here, features are rows, 1 col, 1 channel
        cnn_model.add(Convolution2D(32, (3, 1), input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3]), activation='relu', padding='same')) # Use (3,1) filter for 1D features
        cnn_model.add(MaxPooling2D(pool_size=(2, 1), padding='same')) # Pool across features, not channels
        cnn_model.add(Convolution2D(32, (3, 1), activation='relu', padding='same'))
        cnn_model.add(MaxPooling2D(pool_size=(2, 1), padding='same'))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(units=256, activation='relu'))
        cnn_model.add(Dense(units=y_train_cnn.shape[1], activation='softmax'))
        
        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        text.insert(END, "Starting CNN training (this may take a while)...\n")
        main.update_idletasks() # Update GUI to show message

        hist = cnn_model.fit(X_train_cnn, y_train_cnn, batch_size=16, epochs=100, shuffle=True, verbose=0, validation_data=(X_test_cnn, y_test_cnn))
        
        cnn_model.save_weights('model/cnn_model_weights.h5')
        model_json = cnn_model.to_json()
        with open("model/cnn_model.json", "w") as json_file:
            json_file.write(model_json)
        
        with open('model/cnn_history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
        
        cnn = cnn_model
        text.insert(END, "CNN model training complete and saved.\n\n")

    if cnn:
        text.insert(END, "CNN Model Summary:\n")
        cnn.summary(print_fn=lambda x: text.insert(END, x + '\n'))
        
        text.insert(END, "\nPerforming CNN prediction and metric calculation...\n")
        predict = cnn.predict(X_test_cnn)
        predict = np.argmax(predict, axis=1)
        testY = np.argmax(y_test_cnn, axis=1)
        calculateMetrics("CNN", predict, testY)
    else:
        text.insert(END, "CNN model could not be loaded or trained.\n")


def graph():
    """
    Function to plot the training accuracy and loss graphs for LSTM and CNN models.
    It loads the history from pickle files and displays the plots.
    """
    try:
        # Load LSTM history
        with open('model/lstm_history.pckl', 'rb') as f:
            lstm_data = pickle.load(f)
        lstm_accuracy = lstm_data.get('accuracy', [])
        lstm_loss = lstm_data.get('loss', [])

        # Load CNN history
        with open('model/cnn_history.pckl', 'rb') as f:
            cnn_data = pickle.load(f)
        cnn_accuracy = cnn_data.get('accuracy', [])
        cnn_loss = cnn_data.get('loss', [])

        if not lstm_accuracy and not cnn_accuracy:
            text.insert(END, "No training history found. Please run LSTM and CNN algorithms first.\n")
            return

        plt.figure(figsize=(12, 7))
        plt.grid(True)
        plt.xlabel('EPOCH', fontsize=12)
        plt.ylabel('Accuracy/Loss', fontsize=12)
        
        if lstm_accuracy:
            plt.plot(lstm_accuracy, color='green', marker='o', linestyle='-', linewidth=2, markersize=4, label='LSTM Accuracy')
            plt.plot(lstm_loss, color='blue', marker='o', linestyle='--', linewidth=2, markersize=4, label='LSTM Loss')
        
        if cnn_accuracy:
            plt.plot(cnn_accuracy, color='orange', marker='x', linestyle='-', linewidth=2, markersize=4, label='CNN Accuracy')
            plt.plot(cnn_loss, color='red', marker='x', linestyle='--', linewidth=2, markersize=4, label='CNN Loss')
        
        plt.legend(loc='upper right', fontsize=10)
        plt.title('LSTM Vs CNN Training Accuracy & Loss Graph', fontsize=16)
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        text.insert(END, "Training history files not found. Please run LSTM and CNN algorithms first.\n")
    except Exception as e:
        text.insert(END, f"Error plotting graphs: {e}\n")


def performanceTable():
    """
    Generates an HTML table summarizing the performance metrics (Accuracy, Precision, Recall, FScore, Sensitivity, Specificity)
    for both LSTM and CNN algorithms on the MIT-BH Dataset.
    Opens the generated HTML file in a web browser.
    """
    if not (len(accuracy) >= 2 and len(precision) >= 2 and len(recall) >= 2 and 
            len(fscore) >= 2 and len(sensitivity) >= 2 and len(specificity) >= 2):
        text.insert(END, "Please run LSTM and CNN algorithms first to generate all performance metrics.\n")
        return

    output = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance Metrics</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
            table {
                width: 80%;
                border-collapse: collapse;
                margin: 25px auto; /* Centered table */
                font-size: 0.9em;
                min-width: 400px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                border-radius: 8px;
                overflow: hidden;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #dddddd;
            }
            thead tr {
                background-color: #009879;
                color: #ffffff;
                text-align: left;
            }
            tbody tr:nth-of-type(even) {
                background-color: #f3f3f3;
            }
            tbody tr:last-of-type {
                border-bottom: 2px solid #009879;
            }
            tbody tr.active-row {
                font-weight: bold;
                color: #009879;
            }
            h2 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
    <h2>Automated Detection of Cardiac Arrhythmia Performance Table</h2>
    <table border=1 align=center>
        <thead>
            <tr>
                <th>Dataset Name</th>
                <th>Algorithm Name</th>
                <th>Accuracy (%)</th>
                <th>Precision (%)</th>
                <th>Recall (%)</th>
                <th>F-Score (%)</th>
                <th>Avg. Sensitivity</th>
                <th>Avg. Specificity</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>MIT-BH Dataset</td>
                <td>LSTM</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
            </tr>
            <tr>
                <td>MIT-BH Dataset</td>
                <td>CNN</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
                <td>{:.2f}</td>
            </tr>
        </tbody>
    </table>
    </body>
    </html>
    """.format(accuracy[0], precision[0], recall[0], fscore[0], sensitivity[0], specificity[0],
               accuracy[1], precision[1], recall[1], fscore[1], sensitivity[1], specificity[1])

    try:
        with open("output.html", "w") as f:
            f.write(output)
        webbrowser.open("output.html", new=1)
    except Exception as e:
        text.insert(END, f"Error generating or opening performance table: {e}\n")


def close():
    """
    Function to close the Tkinter application.
    """
    main.destroy()

# GUI Components Setup
font = ('times', 18, 'bold')
title = Label(main, text='Automated Detection of Cardiac Arrhythmia using Recurrent Neural Network')
title.config(bg='DarkGoldenrod1', fg='black')
title.config(font=font)
title.config(height=2, width=80)
title.place(x=0, y=5, relwidth=1)

font1 = ('times', 13, 'bold')

# Buttons for various operations
uploadButton = Button(main, text="Upload Arrhythmia Dataset", command=uploadDataset, padx=10, pady=5)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

pathlabel = Label(main, text="No dataset loaded", wraplength=400, justify='left')
pathlabel.config(bg='brown', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=300, y=100, width=500)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, padx=10, pady=5)
preprocessButton.place(x=50, y=150)
preprocessButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM, padx=10, pady=5)
lstmButton.place(x=50, y=200)
lstmButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN, padx=10, pady=5)
cnnButton.place(x=50, y=250)
cnnButton.config(font=font1)

graphButton = Button(main, text="LSTM & CNN Training Graph", command=graph, padx=10, pady=5)
graphButton.place(x=50, y=300)
graphButton.config(font=font1)

ptButton = Button(main, text="Performance Table", command=performanceTable, padx=10, pady=5)
ptButton.place(x=50, y=350)
ptButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close, padx=10, pady=5)
exitButton.place(x=50, y=400)
exitButton.config(font=font1)

# Text widget for displaying output
font_text = ('times', 12, '')
text = Text(main, height=25, width=80, wrap='word')
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
scroll.config(command=text.yview)
scroll.pack(side=RIGHT, fill=Y)
text.place(x=300, y=150)
text.config(font=font_text)


main.config(bg='LightSteelBlue1')
main.mainloop()
