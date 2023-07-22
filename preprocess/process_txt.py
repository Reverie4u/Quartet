import codecs

import numpy as np


def data_read_txt(file_name):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_txt = codecs.open(file_name, 'r', 'utf-8')  # 读取
    list = file_txt.readlines()
    data = []
    label = []
    for str in list:
        if not str.startswith('0,0,0,0'):
            index = str.rfind(',')
            data.append(eval(str.replace('\n', '')[8:index]))
            label.append(eval(str.replace('\n', '')[index + 1:]))
    return data, label


def data_write_txt(file_name, list):
    file_txt = codecs.open(file_name, 'w', 'utf-8')  # 读取
    file_txt.writelines(list)


def process_txt(name, n):
    name1 = "./data/treeset_" + name + ".txt"
    name2 = "./data/treeset_jit1_" + name + ".txt"
    name3 = "./data/treeset_jit2_" + name + ".txt"
    name4 = "./data/treeset_jit3_" + name + ".txt"
    data1, time1 = data_read_txt(name1)
    data2, time2 = data_read_txt(name2)
    data3, time3 = data_read_txt(name3)
    data4, time4 = data_read_txt(name4)
    length = len(data1)
    time_array = np.zeros((length, 4))
    time_array[:, 0] = np.array(time1)
    time_array[:, 1] = np.array(time2)
    time_array[:, 2] = np.array(time3)
    time_array[:, 3] = np.array(time4)
    label = list(np.argmin(time_array, axis=1))
    data = data1
    data_write_txt("./data/" + name + ".txt", str(data))
    data_write_txt("./data/" + name + "_label.txt", str(label))
    print(len(data))
    print(label)


if __name__ == '__main__':
    # n = 6191
    # process_csv("stack",n)
    # # process_csv("tpch10",22)
    process_txt("tpch", 22)
    # process_csv("job", 113)
