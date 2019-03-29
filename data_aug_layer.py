import caffe
import numpy as np
import random
import pdb
import transformations as trans
import illuminants as illu
import cv2

class DataAugLayer(caffe.Layer):

    def setup(self, bottom, top):

        # config
        random.seed()

        params = eval(self.param_str)
        train_list_path = params['train_list_path']
        self.train_list = np.loadtxt(train_list_path, dtype=str)

        illuminants_path = params['illuminants_path']
        self.illuminants = np.load(illuminants_path)

        self.batch_size = int(params['batch_size'])

        crop_width = int(params['crop_width'])
        crop_height = int(params['crop_height'])
        self.crop_size = (crop_width, crop_height)

        self.sigma = float(params['sigma'])

        self.mean = np.array(params['mean'])
        self.idx = 0

        # comprobar dimensiones illuminants y train_list
        pdb.set_trace()

        np.random.shuffle(self.illuminants)
        np.random.shuffle(self.train_list)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


    def reshape(self, bottom, top):
        
        self.batch = np.zeros()
        for i in np.arange(self.batch_size):

        self.data = illu.apply_illuminant(bottom[0])
        top[0].reshape(*self.data.shape)


    def forward(self, bottom, top):

        top[0].data[...] = self.data
        
        if self.idx == (self.num_illuminants - 1):
            self.idx = 0
            self.illuminants_list = random.shuffle(self.illuminants_list)
        else:
            self.idx += 1


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image_label(self, idx):

        img_path = self.train_list[idx, 0]
        illuminant = self.illuminants[idx]
        label_path = self.train_list[idx, 1]

        white_img = cv2.imread(img_path)
        label = cv2.imread(label_path)[...,0]

        flip_op = (idx % 3) - 1
        cv2.flip(white_img, flip_op)
        cv2.flip(label, flip_op)

        if np.random.rand() < 0.5:
            white_img, top_border = trans.random_crop(white_img, self.crop_size)
            label = trans.crop(label, top_border, self.crop_size)
        else:
            angle = self.sigma * np.random.randn()
            white_img = rotate_img(white_img, angle, self.crop_size)
            label = rotate_img(label, angle, self.crop_size)





        img = illu.apply_illuminant(white_img, illuminant)
        img -= self.mean

        img = img.transpose((2,0,1))

        label_path = self.train_list[idx, 1]
        label = cv2.imread(label_path)
        label = label[...,0]
        label = label[np.newaxis, ...]

        return img


    def load_label(self, idx):

        
        return label
