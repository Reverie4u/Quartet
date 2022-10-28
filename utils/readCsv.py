import numpy as np


def readcsv(csvname):
    p = csvname
    with open(p, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=",")
    return data


if __name__ == '__main__':
    data = readcsv("test.csv")
    print(data)
