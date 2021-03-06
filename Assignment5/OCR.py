import scipy as sp
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sklearn import datasets, svm, metrics
import pickle
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.filters import sobel
from skimage import feature, color
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from enum import Enum

path = './chars74k-lite/'
preprocessing = Enum('Preprocessing', 'SOBEL, HOG, BOTH')
classifiers = Enum('Classifier', 'SVM, KNN, RF')

preprocess = preprocessing.HOG
classifier = classifiers.SVM
detect = True


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


def reshapeImage(image):
    reshaped = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            reshaped.append(image[i][j])
    return reshaped


def trainClassifierSVM(dataSet):
    if os.path.isfile("classifierSVM"):
        file = open("classifierSVM", 'rb')
        classifier = pickle.load(file)
        file.close()
    else:
        classifier = svm.SVC(gamma=10, C=30, probability=True)
        if type(dataSet.images[0][0]) is not np.float64:
            classifier.fit(reshape(dataSet.images), dataSet.letters)
        else:
            classifier.fit(dataSet.images, dataSet.letters)
        file = open("classifierSVM","wb")
        pickle.dump(classifier, file)
        file.close()
    return classifier


def trainClassifierRF(dataSet):
    classifier = RandomForestClassifier()
    if type(dataSet.images[0][0]) is not np.float64:
        classifier.fit(reshape(dataSet.images), dataSet.letters)
    else:
        classifier.fit(dataSet.images, dataSet.letters)
    return classifier


def trainClassifierkNN(dataSet):
    if os.path.isfile("classifier"):
        file = open("classifier", 'rb')
        classifier = pickle.load(file)
        file.close()
    else:
        classifier = KNeighborsClassifier(n_neighbors=5)
        if type(dataSet.images[0][0]) is not np.float64:
            classifier.fit(reshape(dataSet.images), dataSet.letters)
        else:
            classifier.fit(dataSet.images, dataSet.letters)
        file = open("classifier","wb")
        pickle.dump(classifier, file)
        file.close()
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


def detection(filename, classified):
    pixels = 8
    threshold = 0.6
    searchIndex = 13
    origImg = sp.misc.imread(filename)
    origImg /= 256**4
    origImg = np.array(origImg, dtype=np.uint8) # This line only change the type, not values
    img = color.rgb2gray(origImg)
    potential = 0
    detX = []
    detY = []
    sizes = []
    probss = []
    for size in [20, 40, 60]:
        subImages, originals = getSubImages(img, pixels, size)
        print("Number of images", len(subImages)*len(subImages[0]))
        print("Start detecting")
        progress = 0
        quarter = 1
        for i in range(len(subImages)):
            if i >= quarter*len(subImages)/4:
                quarter += 1
                progress += 25
                print("Progess", progress, "%")
            for j in range(len(subImages[i])):
                probs = classified.predict_proba(subImages[i][j])
                maxV = max(probs[0])
                if maxV > threshold:
                    # print(probs[0])
                    for k in range(len(probs[0])):
                        if probs[0][k] == maxV:
                            index = k
                    # printImage(originals[i][j])
                    # printImage(origImg)
                    if index == searchIndex:
                        potential += 1
                        detX.append(i*pixels)
                        detY.append(j*pixels)
                        sizes.append(size)
                        probss.append(probs)
    for i in range(len(detX)):
        drawSquare(origImg, detX[i], detY[i], sizes[i], probss[i], searchIndex)
    print("I think I found", potential, "cases of you letter")
    printImage(img)
    return origImg


def drawSquare(img, pixelX, pixelY, size, probs, index):
    for k in range(size):
        for l in range(size):
            if k < 3 or k > size-4 or l < 3 or l > size-4:
                img[pixelX+k][pixelY+l] = [255*(1-2*probs[0][index]), 255*probs[0][index], 0]


def getSubImages(img, pixels, size):
    subImages = []
    originals = []
    for i in range(len(img)):
        subImageRow = []
        originalRow = []
        for j in range(len(img[i])):
            if i % pixels == 0 and j % pixels == 0 and i+size-1 < len(img) and j+size-1 < len(img[i]):
                subImage = []
                for k in range(i, i+size, int(size/20)):
                    line = []
                    for l in range(j, j+size, int(size/20)):
                        line.append(img[k][l])
                    subImage.append(line)
                originalRow.append(subImage)
                if preprocess == preprocessing.SOBEL:
                    subImage = denoise_bilateral(subImage, sigma_range=0.1, sigma_spatial=15)
                    subImage = sobel(subImage)
                elif preprocess == preprocessing.HOG:
                    subImage = useHoG(subImage)
                else:
                    subImage = denoise_bilateral(subImage, sigma_range=0.1, sigma_spatial=15)
                    subImage = sobel(subImage)
                    subImage = useHoG(subImage)
                subImageRow.append(subImage)
        if len(subImageRow) > 0:
            subImages.append(subImageRow)
            originals.append(originalRow)
    return subImages, originals


def findParameters():
    dataSet = loadImages()
    dataSet, testSet = splitDataset(0.1, dataSet)
    best = 0
    bestC = 0
    bestGamma = 0
    for i in [0.1, 1, 2, 5, 10, 20, 30]:
        for j in [0.1, 1, 2, 5, 10, 20, 50, 100]:
            classifier = svm.SVC(gamma=i, C=j)
            if type(dataSet.images[0][0]) is not np.float64:
                classifier.fit(reshape(dataSet.images), dataSet.letters)
            else:
                classifier.fit(dataSet.images, dataSet.letters)
            if type(testSet.images[0][0]) is not np.float64:
                predicted = classifier.predict(reshape(testSet.images))
            else:
                predicted = classifier.predict(testSet.images)
            correct = 0
            false = 0
            images_and_predictions = list(zip(testSet.images, predicted))
            for index, (image, prediction) in enumerate(images_and_predictions):
                if prediction == testSet.letters[index]:
                    correct += 1
                else:
                    false += 1
            if correct/(correct+false) > best:
                best = correct/(correct+false)
                print("Best so far:", best)
                bestC = j
                bestGamma = i
    print("Best c:", bestC)
    print("Best gamma:", bestGamma)
    print("Best prediction:", best)


def testMain():
    avg = 0
    for i in range(5):
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
        avg += correct/(correct+false)
    print("Avg:", avg/5)


def main():
    dataSet = loadImages()
    dataSet, testSet = splitDataset(0.1, dataSet)
    if classifier == classifiers.SVM:
        classified = trainClassifierSVM(dataSet)
    elif classifier == classifiers.KNN:
        classified = trainClassifierkNN(dataSet)
    elif classifier == classifiers.RF:
        classified = trainClassifierRF(dataSet)
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
    # showImages(dataSet, testSet, predicted)
    # myTest(classified, 'testD.jpg', 'd')
    if detect:
        sp.misc.imsave('detectedTest.jpg', detection('handwritten.jpg', classified))
        # sp.misc.imsave('detectedTestImg.jpg', detection('testImg.jpg', classified))

main()
# testMain()
# findParameters()