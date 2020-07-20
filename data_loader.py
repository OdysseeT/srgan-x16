#import scipy
from glob import glob
import numpy as np
import os
import cv2
#import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, datadir, dataset_name, scale, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.datadir = datadir
        self.scale = scale

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob("{}/{}/*".format(self.datadir, self.dataset_name))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:

            if not os.path.isfile("./data/{}_HR/{}".format(self.dataset_name, os.path.basename(img_path))):
                continue
                
            img_hr = cv2.imread("{}/{}_HR/{}".format(self.datadir, self.dataset_name, os.path.basename(img_path)))
            img_lr = cv2.imread("{}/{}_LR/{}".format(self.datadir, self.dataset_name, os.path.basename(img_path)))

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    def preprocess(self):

        path = glob('./data/%s/*' % (self.dataset_name))

        if os.path.isfile("./data/{}_HR/{}".format(self.dataset_name, os.path.basename(path[0]))):
            print("Preprocessing already done.")
            return

        h, w = self.img_res
        low_h, low_w = int(h / self.scale), int(w / self.scale)

        for img_path in path:

            print("Transforming ", img_path)
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB).astype(np.float)

            img_hr = cv2.resize(img, self.img_res)
            img_lr = cv2.resize(img, (low_h, low_w))

            # If training => do random flip
            if np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            cv2.imwrite("./data/{}_HR/{}".format(self.dataset_name, os.path.basename(img_path)), img_hr)
            cv2.imwrite("./data/{}_LR/{}".format(self.dataset_name, os.path.basename(img_path)), img_lr)

    #def imread(self, path):
    #    return scipy.misc.imread(path, mode='RGB').astype(np.float)
