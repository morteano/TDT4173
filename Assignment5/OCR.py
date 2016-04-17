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
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
from skimage import img_as_bool
from skimage.morphology import black_tophat, skeletonize, convex_hull_image

path = './chars74k-lite/'


class ImagePairs:
    def __init__(self):
        self.images = []
        self.letters = []


def loadImages():
    # Creates an initial imagePair
    imagePairs = ImagePairs()

    # Append the images and the correct labels to imagesPairs
    for directory in os.listdir(path):
        if len(directory) < 2:
            for filename in os.listdir(path+directory+'/'):
                img = sp.misc.imread(path+directory+'/'+filename)
                img = img.astype(np.float)/255
                # printImage(img)
                img = denoise_bilateral(img, sigma_range=0.1, sigma_spatial=15)
                # printImage(img)
                img = feature.canny(img)
                # printImage(img)
                imagePairs.images.append(img)
                imagePairs.letters.append(directory)
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


def main():
    dataSet = loadImages()
    dataSet, testSet = splitDataset(0.1, dataSet)
    # images_and_labels = list(zip(dataSet.images, dataSet.letters))
    # for index, (image, label) in enumerate(images_and_labels[:4]):
    #     plt.subplot(2, 4, index + 1)
    #     plt.axis('off')
    #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title('Training: ' + label)
    classifier = svm.SVC()
    classifier.fit(reshape(dataSet.images), dataSet.letters)
    predicted = classifier.predict(reshape(testSet.images))
    images_and_predictions = list(zip(testSet.images, predicted))
    correct = 0
    false = 0
    for index, (image, prediction) in enumerate(images_and_predictions):
        if prediction == testSet.letters[index]:
            correct += 1
            print("Correct:", prediction, testSet.letters[index])
        else:
            false += 1
            print("False:", prediction, testSet.letters[index])
    print(correct/(correct+false))

    # plt.subplot(2, 4, index + 5)
    # plt.axis('off')
    # plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.title('Prediction: ' + prediction)
    # print(testSet.letters[index])
    # plt.show()

main()



# /255, denoise(0.05,15), sobel -> 37.8 %


