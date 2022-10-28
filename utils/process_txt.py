import codecs


def read_txt(file_name):
    file_txt = codecs.open(file_name, 'r', 'utf-8')
    list = file_txt.readline()
    return eval(list)


def write_txt(file_name, data):
    # data is a list
    file_txt = codecs.open(file_name, 'w', 'utf-8')
    file_txt.writelines(str(data))


if __name__ == '__main__':
    pass
