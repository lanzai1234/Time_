import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os


def load_images(path):
    img = Image.open(path)
    width = int(512)
    height = int(512)
    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img)
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    image_r_normalization = (img_r - 0) / 255
    image_g_normalization = (img_g - 0) / 255
    image_b_normalization = (img_b - 0) / 255
    image_r_normalization = image_r_normalization.reshape((512, 512, 1))
    image_g_normalization = image_g_normalization.reshape((512, 512, 1))
    image_b_normalization = image_b_normalization.reshape((512, 512, 1))
    image_normalization = np.concatenate((image_r_normalization, image_g_normalization, image_b_normalization), axis=2)
    return image_normalization


def load_labels(index):
    labels_one_hot = np.array([
        [[0], [0], [0], [0], [1]],
        [[0], [0], [0], [1], [0]],
        [[0], [0], [1], [0], [0]],
        [[0], [1], [0], [0], [0]],
        [[1], [0], [0], [0], [0]]
    ])
    labels = np.array([[0],[1],[2],[3],[4]])
    return labels[index]


def walkFile(root_path):
    for root, dirs, files in os.walk(root_path):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历所有的文件夹
        label_index = 0
        labels = []
        image_array = []
        for d in dirs:
            sub_path = os.path.join(root, d)
            for sub_root, sub_dirs, sub_files in os.walk(sub_path):
                for f in sub_files:
                    labels.append(load_labels(label_index))
                    img_path = os.path.join(sub_root, f)
                    img = load_images(img_path)
                    image_array.append(img)
            label_index = label_index + 1
        labels = np.array(labels)
        return image_array, labels


if __name__ == "__main__":
    #walkFile("G:\牛梦毫_zy1906134_医疗影像计算大作业\mycode\dataset\\train")
    load_images("\dataset\\train")