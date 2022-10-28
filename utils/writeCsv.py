import numpy as np


def writecsv(csvname, data):
    p = csvname
    with open(p, 'w', encoding='utf-8') as f:
        np.savetxt(f, data, delimiter=",")


if __name__ == '__main__':
    test = np.array([[1, 2, 3], [4, 5, 6]])
    writecsv("test.csv", test)
