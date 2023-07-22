import numpy as np

import utils.readCsv as readCsv
import utils.writeCsv as writeCsv


def process_csv(name, n):
    name1 = "./data/dataset_" + name + ".csv"
    name2 = "./data/dataset_jit1_" + name + ".csv"
    name3 = "./data/dataset_jit2_" + name + ".csv"
    name4 = "./data/dataset_jit3_" + name + ".csv"
    data1 = readCsv.readcsv(name1)
    data2 = readCsv.readcsv(name2)
    data3 = readCsv.readcsv(name3)
    data4 = readCsv.readcsv(name4)
    # 删除非select语句
    data1 = np.delete(data1, np.where(data1[:, 3] == 0), axis=0)
    data2 = np.delete(data2, np.where(data2[:, 3] == 0), axis=0)
    data3 = np.delete(data3, np.where(data3[:, 3] == 0), axis=0)
    data4 = np.delete(data4, np.where(data4[:, 3] == 0), axis=0)
    length, width = data1.shape
    time_array = np.zeros((length, 4))
    time_array[:, 0] = data1[:, width - 1]
    time_array[:, 1] = data2[:, width - 1]
    time_array[:, 2] = data3[:, width - 1]
    time_array[:, 3] = data4[:, width - 1]
    label = np.argmin(time_array, axis=1)
    data = data1[:, :width - 1]
    print(label)
    # 过滤出特征列
    idx1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 18, 20, 21, 22, 23, 24, 25, 26, 27, 33, 35, 36, 37, 38, 40,
            41, 42]
    data = np.delete(data, idx1, axis=1)
    # idx2用于去除重复行
    _, idx2 = np.unique(data, axis=0, return_index=True)
    idx2 = np.sort(idx2)
    data = data[idx2]
    label = label[idx2]
    print(label)
    print(data.shape)
    writeCsv.writecsv("./data/" + name + ".csv", data)
    writeCsv.writecsv("./data/" + name + "_label.csv", label)


if __name__ == '__main__':
    process_csv("tpch", 22)
