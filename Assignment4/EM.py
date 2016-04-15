import math
import matplotlib.pyplot as plt


def getData():
    with open("sample-data.txt") as f:
        data = []
        for line in f:
            data.append(float(line))
    return data


def pdf(x, mean, sigma):
    return math.exp(-(x-mean)**2/(2*sigma**2))/sigma*math.sqrt(2*math.pi)


def E(data, means, i, j):
    numerator = pdf(data[i], means[j], 1)
    denomerator = 0
    for n in range(len(means)):
        denomerator += pdf(data[i], means[n], 1)
    return numerator/denomerator


def M(expected, x, j):
    numerator = 0
    for i in range(len(x)):
        numerator += expected[j][i]*x[i]
    return numerator/sum(expected[j])


def main():
    data = getData()
    alpha = sorted(data)
    mid = int(len(data)/2)
    gaussians = [[], []]
    for i in range(mid):
        gaussians[0].append(alpha[i])
        gaussians[1].append(alpha[i+mid])
    means = [sum(gaussians[0])/len(gaussians[0]), sum(gaussians[1])/len(gaussians[1])]

    for iteration in range(100):
        if (iteration == 5) or (iteration == 10):
            print("Iteration nr " + str(iteration)+": ", means)

        expected = []
        for j in range(len(means)):
            line = []
            for i in range(len(data)):
                line.append(E(data, means, i, j))
            expected.append(line)

        newMeans = []
        for j in range(len(means)):
            newMeans.append(M(expected, data, j))
        means = newMeans
    print("Iteration nr " + str(iteration)+": ", means)
    plt.hist(data)
    plt.axvline(means[0], linewidth=5, color='r')
    plt.axvline(means[1], linewidth=5, color = 'g')
    plt.show()


main()