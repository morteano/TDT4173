import matplotlib.pyplot as plt

TESTSET = "data/data-test.csv"
TRAINSET = "data/data-train.csv"


# stores the csv file as a matrix
def getFileAsMatrix(filename):
    file = open(filename, "rb")
    i = 0
    data = []
    for line in file.readlines():
        dataLine = []
        for number in line.split(","):
            dataLine.append(float(number))
        data.append(dataLine)
    return data

# splits a matrix into x values and y values
def splitData(data):
    x = []
    y = []
    for line in data:
        x.append([line[0], line[1]])
        y.append(line[2])
    return x, y


def f(x, b, W):
    result = b
    for j in range(len(W)):
        result += W[j]*x[j]
    return result


def L(x, y, b, W):
    result = 0
    for i in range(len(x)):
        result += (f(x[i], b, W)-y[i])**2
    return result/len(x)


def dLdW(x, y, b, W):
    result = [0, 0]
    n = len(x)
    for i in range(n):
        result[0] += (f(x[i], b, W)-y[i])*x[i][0]
        result[1] += (f(x[i], b, W)-y[i])*x[i][1]
    return [result[0]*2/n, result[1]*2/n]


def dLdb(x, y, b, W):
    result = 0
    n = len(x)
    for i in range(n):
        result += f(x[i], b, W)-y[i]
    return result*2/n


def main():
    alpha = 0.01
    b = 0
    W = [0, 0]
    data = getFileAsMatrix(TRAINSET)
    x, y = splitData(data)
    maxIterations = 100

    testData = getFileAsMatrix(TESTSET)
    x_test, y_test = splitData(testData)

    loss = []
    i = 0
    while i < maxIterations:
        if i > 1:
            if abs(loss[i-1] - loss[i-2]) < 0.00001:
                break
        [dW1, dW2] = dLdW(x, y, b, W)
        W = [W[0]-alpha*dW1, W[1]-alpha*dW2]
        b -= alpha*dLdb(x, y, b, W)
        loss.append(L(x_test, y_test, b, W))
        i += 1

    print(loss[4])
    print(loss[9])
    print(W)
    print(b)
    plt.plot(range(0,len(loss)), loss)
    plt.show()

if __name__ == "__main__":
    main()