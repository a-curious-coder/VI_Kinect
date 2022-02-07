
from argparse import ArgumentParser
from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper

import os, os.path
import io

import cv2
import time
import numpy as np
from pickle import dump
from pickle import load

import seaborn as sns
import matplotlib.pyplot as plt
from PIL  import Image
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import confusion_matrix as cf

# TRAINING OR TESTING
# MODE = "train"
# MODE = "predict"
MODE = "confusion"

SHOW_EIGENFACES = True

NAMES = []
N_COMPONENTS = 150


# OR: clear = lambda: os.system('clear')
if os.name == 'nt':
    clear = lambda: os.system('cls')    # Used when printing information to terminal
else:
    clear = lambda: os.system('clear') 



def plot_eigenfaces(images,  h, w, n_row=3, n_col=4):

    titles = ["eigenface %d" % i for i in range(images.shape[0])]

    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def train(X_train, y_train):
    """ creates a classifier model from eigenfaces """
    n_samples, h, w = X_train.shape
    # 3D Array to 2D
    print("Reshaping array")
    X_train = X_train.reshape((X_train.shape[0], -1))

    # PCA FIT
    print("Fitting data to PCA")
    pca = PCA(n_components=N_COMPONENTS, svd_solver='randomized', whiten=True).fit(X_train)

    if SHOW_EIGENFACES:
        print("Creating eigenfaces")
        eigenfaces = pca.components_.reshape((N_COMPONENTS, h, w))


        plot_eigenfaces(eigenfaces, h, w)
        plt.show()


    print("Transform pca using x_train")
    X_train_pca = pca.transform(X_train)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    
    
    clf = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'), param_grid
    )

    print("Fit classifier to PCA data")
    # Fit pca to classifier data
    clf = clf.fit(X_train_pca, y_train)

    # Returns classification and pca results
    return clf, pca


def predict_video(model, pca, names,  videofolder, no_realtime):
    with open("predictions.txt", "w") as f:
        f.close()
    predictions = []

    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (10, 50) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (0, 255, 0) 
    # Line thickness of 2 px 
    thickness = 2

    total_to_do = len([name for name in os.listdir(videofolder) if name.endswith(".pgm")])
    temp_pred = 0
    count = 0
    mean = 3.1
    for status, rgb, depth in FreenectPlaybackWrapper(videofolder, not no_realtime):
        # If we have an updated Depth image, then add to images
        if status.updated_depth: 
            temp_pred = predict_frame(model, pca, depth)
            ret, thresh_depth = cv2.threshold(depth, 110, 254, cv2.THRESH_TOZERO)

            mean = np.mean(thresh_depth)
            # Using cv2.putText() method 
            if mean <= 3.5:
                depth = cv2.putText(depth, "Nothing", org, font,  fontScale, color, thickness, cv2.LINE_AA)
            else:
                depth = cv2.putText(depth, print_label(temp_pred), org, font,  fontScale, color, thickness, cv2.LINE_AA)
            
            
            
            with open("predictions.txt", "a") as f:
                f.write(f"{int(temp_pred[0])}\n")

            cv2.imshow("Depth", depth)
            # print_labels(temp_pred, names)
            count += 1

            predictions.append(temp_pred)
        if status.updated_rgb:
            if mean <= 3.5:
                rgb = cv2.putText(rgb, "Nothing", org, font,  fontScale, color, thickness, cv2.LINE_AA)
            else:
                rgb = cv2.putText(rgb, print_label(temp_pred), org, font,  fontScale, color, thickness, cv2.LINE_AA)
            
            cv2.imshow("RGB", rgb)
        print(f"Frame: {count} / {total_to_do}")
        if cv2.waitKey(10) == 27:
            break



    return predictions


def predict_frame(model, pca, frame):
    ''' Predict objects using provided data'''
    # Converts the 3D array to 2D (Just the image dimensions)
    X_test = frame.reshape((frame.shape[0], -1))
    # Converts the image array to a 1D array
    X_test = X_test.flatten()
    # Need to convert back to a 2D array with a format like: (1, 372000) 
    X_test = X_test.reshape(1, -1)

    X_test_pca = pca.transform(X_test)
    
    y_pred = model.predict(X_test_pca)

    return y_pred


def train_from_video(model, pca, videofolder, no_realtime):
    ''' Creates a numpy array from the video input '''
    for status, rgb, depth in FreenectPlaybackWrapper(videofolder, not no_realtime):
        h, w, z = depth.shape   # Run through once to get dimensions of images
        break

    # print("Got dimension of images")
    # print("h: " + str(h) + " w: " + str(w))

    count = 0  # number of images traversed so far
    current_class = 0
    # Assuming that the set number of images will remain the same - 14
    total_to_do = len([name for name in os.listdir(videofolder) if name.endswith(".pgm")])     # Find number of frames in source folder
    batch_size = total_to_do//14
    # Initialise images and labels size to accommodate for all images.
    images = np.zeros((total_to_do, h, w), dtype = np.uint8)
    labels = np.zeros((total_to_do))

    for status, rgb, depth in FreenectPlaybackWrapper(videofolder, not no_realtime):
        # If we have an updated Depth image/frame, then add it to images array
        if status.updated_depth: 
            depth = depth.reshape((1, depth.shape[0],  depth.shape[1]))          # Reshape to add


            # Overwrite zero value with depth image
            images[count] = depth
            # Overwrite zero value with current_class value (represents the object we're assuming is in frame)
            labels[count] = current_class

            

            clear() 
            # Print out stats for progress
            percentage = format(((count+1) / total_to_do) * 100, ".2f")
            print(f"Creating images {percentage}%")
            print(f"Image: {count+1}/{total_to_do}")
            print(f"Class: {current_class}")

            # Iterate the object number (current_class)
            if count % batch_size == 0 and count != 0 and current_class < 13:
                current_class += 1

            count += 1

    print("Starting training on images")
    start = time.time()
    # Creates a model and pca based on images/labels accumulated so far
    model, pca = train(images, labels)
    end = time.time()
    print(f"Model and PCA took {end-start} seconds")
    
    # Images and labels arrays will be unloaded from memory once this function has finished
    return model, pca

def load_label_names():
    """
    Used to load the names into a variable
    Been put into a function so it's only used once (reduces slow IO calls)
    """
    with open('Set1Labels.txt', 'r') as file:
        return file.read().splitlines()

def print_label(label):
    clear()
    return NAMES[int(label)]

def print_labels(labels, names):
    """
    Is used to take the prediction result lables and predefined labels to 
    show the output with labels
    """
    # SET TRUE IF YOU WANT OBJECTS TO BE PRINTED ONCE
    print_once = True
    # Label indexes to object names
    label_names = []
    object_appended = False
    p_label = 0
    # print("Labels")
    
    for label in labels:
        if p_label == label and object_appended and print_once:
            continue
        p_label = label
        object_appended = False
        # print(label, lines[label])
        label_names.append(names[int(label)])
        object_appended = True
    clear()
    print(label_names)


def main():
    global N_COMPONENTS
    global MODE
    global NAMES

    parser = ArgumentParser(description="OpenCV Demo for Kinect Coursework")
    parser.add_argument("videofolder", help="Folder containing Kinect video. Folder must contain INDEX.txt.",
                        default="ExampleVideo", nargs="?")
    parser.add_argument("--no-realtime", action="store_true", default=True)

    args = parser.parse_args()
    # Print arguments for debug
    print(args)
    NAMES = load_label_names()

    assert MODE == "train" or MODE == "predict" or MODE == "confusion"
    
    if MODE == "train":
        print()
        print("Creating model")

        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        pca = PCA(n_components=N_COMPONENTS, svd_solver='randomized',
            whiten=True)

        model = GridSearchCV(
            SVC(kernel='rbf', class_weight='balanced'), param_grid
        )

        print("[*] Creating dataset")
        model, pca = train_from_video(model, pca, args.videofolder, args.no_realtime)

        dump(model, open("MODEL.pkl", 'wb'))
        dump(pca, open("PCA.pkl", 'wb'))

        print("[*] Training model: DONE")

    elif MODE == "predict":

        print("[*] Loading labels")
        names = load_label_names()
        
        # print("[*] Loading models")
        pca = load(open('PCA.pkl', 'rb'))
        model = load(open('MODEL.pkl', 'rb'))
        print("[*] Starting predictions")
        results = predict_video(model, pca, names, args.videofolder, args.no_realtime)

        print("[*] Final results")

        print(print_labels(results, NAMES))
        # print_labels(results)
    elif MODE == "confusion":
        total_to_do = len([name for name in os.listdir(args.videofolder) if name.endswith(".pgm")])

        for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):
            h, w, z = depth.shape   # Run through once to get dimensions of images
            break
        actual_labels = np.zeros((total_to_do))
        each_object = total_to_do//14
        object_counter = 0
        count = 0;

        print(total_to_do)
        for i in range(0, 14):
            for j in range(each_object*i, each_object*(i+1)):
                actual_labels[j] = int(object_counter)
            object_counter += 1

        if count < total_to_do and i == 13:
            for i in range(each_object*(i+1), total_to_do):
                actual_labels[i] = int(object_counter-1)

        with open("actual.txt", "w") as f:
            for i in actual_labels:
                f.write(f"{int(i)}\n")

        with open("predictions.txt", "r") as f:
            p = f.read().splitlines()

        with open("actual.txt", "r") as f:
            a = f.read().splitlines()

        cm = confusion_matrix(a, p)
        # sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues', cbar = False)
        cf.make_confusion_matrix(cm, categories=NAMES, count = False,  figsize = (15, 7))
        plt.show()
    


    return 0

if __name__ == "__main__":
    exit(main())