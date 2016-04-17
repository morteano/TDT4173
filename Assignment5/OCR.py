import scipy as sp
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn import datasets, svm, metrics
from scipy import ndimage
import PIL
import time
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.filters import sobel
from skimage import feature, color
from sklearn.neighbors import KNeighborsClassifier
from enum import Enum

path = './chars74k-lite/'
preprocessing = Enum('Preprocessing', 'SOBEL, HOG, BOTH')
classifiers = Enum('Classifier', 'SVM, KNN')

preprocess = preprocessing.BOTH
classifier = classifiers.KNN


class ImagePairs:
    def __init__(self):
        self.images = []
        self.letters = []
        self.originals = []


def loadImages():
    # Creates an initial imagePair
    imagePairs = ImagePairs()
    # Append the images and the correct labels to imagesPairs
    for directory in os.listdir(path):
        if len(directory) < 2:
            for filename in os.listdir(path+directory+'/'):
                img = sp.misc.imread(path+directory+'/'+filename)
                imagePairs.originals.append(img)
                if preprocess == preprocessing.SOBEL:
                    img = useSOBEL(img)
                elif preprocess == preprocessing.HOG:
                    img = useHoG(img)
                else:
                    img = useSOBEL(img)
                    img = useHoG(img)
                imagePairs.images.append(img)
                imagePairs.letters.append(directory)
    return imagePairs


def useSOBEL(img):
    img = img.astype(np.float)/255
    img = denoise_bilateral(img, sigma_range=0.1, sigma_spatial=15)
    img = sobel(img)
    return img


def useHoG(img):
    img = feature.hog(img, orientations=9, pixels_per_cell=(3, 3), cells_per_block=(6, 6))
    return img


def loadTestImages(filename, letter):
    # Creates an initial imagePair
    imagePairs = ImagePairs()
    # Append the images and the correct labels to imagesPairs
    img = sp.misc.imread(filename)
    imagePairs.originals.append(img)
    if preprocess == preprocessing.SOBEL:
        img = useSOBEL(img)
    elif preprocess == preprocessing.HOG:
        img = useHoG(img)
    else:
        img = useSOBEL(img)
        img = useHoG(img)
    imagePairs.images.append(img)
    imagePairs.letters.append(letter)
    return imagePairs


def printImage(img):
    plt.figure()
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


def splitDataset(percentage, dataset):
    testSet = ImagePairs()
    testSetSize = int(len(dataset.images)*percentage)
    for i in range(testSetSize):
        index = random.randint(0, len(dataset.images)-1)
        testSet.images.append(dataset.images.pop(index))
        testSet.letters.append(dataset.letters.pop(index))
        testSet.originals.append(dataset.originals.pop(index))
    return dataset, testSet


def reshape(images):
    reshapedList = []
    for i in range(len(images)):
        reshaped = []
        for j in range(len(images[i])):
            for k in range(len(images[i][j])):
                reshaped.append(images[i][j][k])
        reshapedList.append(reshaped)
    return reshapedList


def trainClassifierSVM(dataSet):
    classifier = svm.SVC(probability=True)
    if type(dataSet.images[0][0]) is not np.float64:
        classifier.fit(reshape(dataSet.images), dataSet.letters)
    else:
        classifier.fit(dataSet.images, dataSet.letters)
    return classifier


def trainClassifierkNN(dataSet):
    classifier = KNeighborsClassifier(n_neighbors=5)
    if type(dataSet.images[0][0]) is not np.float64:
        classifier.fit(reshape(dataSet.images), dataSet.letters)
    else:
        classifier.fit(dataSet.images, dataSet.letters)
    return classifier


def predictClassifier(classifier, testSet):
    if type(testSet.images[0][0]) is not np.float64:
        return classifier.predict(reshape(testSet.images))
    else:
        return classifier.predict(testSet.images)


def showImages(dataSet, testSet, predicted):
    images_and_labels = list(zip(dataSet.originals, dataSet.letters))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: ' + label)
    images_and_predictions = list(zip(testSet.originals, predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: ' + prediction)
    plt.show()


def myTest(classified, filename, letter):
    testSet = loadTestImages(filename, letter)
    predicted = predictClassifier(classified, testSet)
    correct = 0
    false = 0
    images_and_predictions = list(zip(testSet.images, predicted))
    for index, (image, prediction) in enumerate(images_and_predictions):
        if prediction == testSet.letters[index]:
            correct += 1
            print("Correct:", prediction, testSet.letters[index])
        else:
            false += 1
            print("False:", prediction, testSet.letters[index])


def detection(filename):
    img = sp.misc.imread(filename)
    img = color.rgb2gray(img)
    subImages = getSubImages(img)
    print(len(subImages))
    return img


def getSubImages(img):
    subImages = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            if i % 5 == 4 and j % 5 == 4 and i + 9 < len(img) and j+9 < len(img[i]):
                subImage = []
                for k in range(i-10, i+9):
                    line = []
                    for l in range(j-10, j+9):
                        line.append(img[k][l])
                    subImage.append(line)
                subImages.append(subImage)
    return subImages


def main():
    dataSet = loadImages()
    dataSet, testSet = splitDataset(0.1, dataSet)
    if classifier == classifiers.SVM:
        classified = trainClassifierSVM(dataSet)
    elif classifier == classifiers.KNN:
        classified = trainClassifierkNN(dataSet)
    predicted = predictClassifier(classified, testSet)
    correct = 0
    false = 0
    images_and_predictions = list(zip(testSet.images, predicted))
    for index, (image, prediction) in enumerate(images_and_predictions):
        if prediction == testSet.letters[index]:
            correct += 1
            print("Correct:", prediction, testSet.letters[index])
        else:
            false += 1
            print("False:", prediction, testSet.letters[index])
    print(correct/(correct+false))
    showImages(dataSet, testSet, predicted)
    myTest(classified, 'testD.jpg', 'd')

# main()
detection('rsz_grades.jpg')
