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


        self.path = glob("{}/{}/*".format(self.datadir, self.dataset_name))
        self.dir_hr = "{}/{}_HR_{}".format(self.datadir, self.dataset_name, self.img_res[0])
        self.dir_lr = "{}/{}_LR_{}".format(self.datadir, self.dataset_name, self.img_res[0])

    def load_data(self, batch_size=1, is_testing=False):
        #data_type = "train" if not is_testing else "test"

        #path = glob("{}/{}/*".format(self.datadir, self.dataset_name))


        imgs_hr = []
        imgs_lr = []

        while len(imgs_hr) < batch_size:

            batch_images = np.random.choice(self.path, size=batch_size)
            for img_path in batch_images:

                if not os.path.isfile("{}/{}".format(self.dir_hr, os.path.basename(img_path))):
                    print("No HR for file {}".format(img_path))
                    continue

                img_hr = cv2.imread("{}/{}".format(self.dir_hr, os.path.basename(img_path)))
                img_lr = cv2.imread("{}/{}".format(self.dir_lr, os.path.basename(img_path)))

                imgs_hr.append(img_hr)
                imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    def preprocess(self):


        if not os.path.exists(self.dir_hr):
            os.makedirs(self.dir_hr)
            os.makedirs(self.dir_lr)
        elif os.path.isfile("{}/{}".format(self.dir_hr, os.path.basename(self.path[0]))):
            print("Preprocessing already done.")
            return

        h, w = self.img_res
        low_h, low_w = int(h / self.scale), int(w / self.scale)

        for img_path in self.path:

            print("Transforming ", img_path)
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB).astype(np.float)

            img_hr = cv2.resize(img, self.img_res)
            img_lr = cv2.resize(img, (low_h, low_w))

            # If training => do random flip
            if np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            cv2.imwrite("{}/{}".format(self.dir_hr, os.path.basename(img_path)), img_hr)
            cv2.imwrite("{}/{}".format(self.dir_lr, os.path.basename(img_path)), img_lr)

    #def imread(self, path):
    #    return scipy.misc.imread(path, mode='RGB').astype(np.float)
