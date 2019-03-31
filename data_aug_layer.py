import caffe
import numpy as np
import random
import pdb
import transformations as trans
import illuminants as illu
import cv2
import matplotlib.pyplot as plt
import vis
import os.path
import shutil

class DataAugLayer(caffe.Layer):

    def setup(self, bottom, top):

        # config
        random.seed()

        params = eval(self.param_str)
        train_list_path = params['train_list_path']
        self.train_list = np.loadtxt(train_list_path, dtype=str)

        illuminants_path = params.get('illuminants_path', None)
        if illuminants_path is not None:
            self.illuminants = np.load(illuminants_path)
        else:
            raise Exception("Illuminants path missing.")

        self.batch_size = int(params['batch_size'])

        crop_width = int(params['crop_width'])
        crop_height = int(params['crop_height'])
        self.crop_size = (crop_width, crop_height)

        self.sigma = float(params['sigma'])

        self.mean = np.array(params['mean'])
        self.idx = 0

        self.testing = params.get('testing', False)


        if self.testing:

            print ">> TESTING:"
            print ">> train_list_path: {}".format(train_list_path)
            print ">> illuminants_path: {}".format(illuminants_path)
            print ">> batch_size: {}".format(self.batch_size)
            print ">> crop_size: {}".format(self.crop_size)
            print ">> sigma: {}".format(self.sigma)
            print ">> mean: {}".format(self.mean)
            print
            #pdb.set_trace()

        np.random.shuffle(self.train_list)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


    def reshape(self, bottom, top):
        

        self.batch_img = np.zeros((self.batch_size, 3, self.crop_size[1], self.crop_size[0]), dtype=np.float32)
        self.batch_label = np.zeros((self.batch_size, 1, self.crop_size[1], self.crop_size[0]), dtype=np.float32)

        for i in np.arange(self.batch_size):

            img_path = self.train_list[self.idx, 0]
            label_path = self.train_list[self.idx, 1]

            if self.testing:
                img_name = os.path.basename(img_path)
                print ">> Image {}: {}".format(i, img_name)

            img, label = self.load_image_label(img_path, label_path)
            self.batch_img[i, ...] = img
            self.batch_label[i, ...] = label

            if self.idx == len(self.train_list) - 1:
                self.idx = 0
                np.random.shuffle(self.train_list)
            else:
                self.idx += 1

        top[0].reshape(*self.batch_img.shape)
        top[1].reshape(*self.batch_label.shape)


    def forward(self, bottom, top):

        top[0].data[...] = self.batch_img
        top[1].data[...] = self.batch_label


        
    def sample_illuminant(self):

        idx_illu = np.random.randint(0, len(self.illuminants)-1)
        illuminant = self.illuminants[idx_illu, :].flatten()
        if self.testing:
            print ">> illu_value: {}".format(illuminant)
        return illuminant


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image_label(self, img_path, label_path):

        white_img = cv2.imread(img_path)
        label = cv2.imread(label_path)[...,0]

        if self.testing:
            plt.figure()
            plt.imshow(white_img[...,::-1])
            plt.title("White Image")

        flip_op = np.random.choice([-1,0,1])
        white_img = cv2.flip(white_img, flip_op)
        label = cv2.flip(label, flip_op)

        if self.testing:
            vis_img = vis.vis_seg(white_img.astype(np.uint8), label, vis.make_palette(20))
            plt.figure()
            plt.imshow(vis_img[...,::-1])
            plt.title("Flipped {} Image".format(flip_op))

        if np.random.rand() < 0.5:

            white_img, top_border = trans.random_crop(white_img, self.crop_size)
            label = trans.crop(label, top_border, self.crop_size)

            if self.testing:
                vis_img = vis.vis_seg(white_img.astype(np.uint8), label, vis.make_palette(20))
                plt.figure()
                plt.imshow(vis_img[...,::-1])
                plt.title("Cropped Image")

        else:

            angle = self.sigma * np.random.randn()
            white_img = trans.rotate_img(white_img, angle, self.crop_size)
            label = trans.rotate_img(label, angle, self.crop_size, is_mask=True)

            if self.testing:
                label_vis = label.copy()
                label_vis[label_vis == 255] = 5
                vis_img = vis.vis_seg(white_img.astype(np.uint8), label_vis, vis.make_palette(20))
                plt.figure()
                plt.imshow(vis_img[...,::-1])
                plt.title("Rotated Image_{}".format(int(angle)))

        illuminant = self.sample_illuminant()
        img = illu.apply_illuminant(white_img, illuminant)

        if self.testing:
            plt.figure()
            plt.imshow(img[...,::-1])
            plt.title("New Image")
            plt.show()
            print 

        img = img.astype(float)
        img -= self.mean
        
        img = img.transpose((2,0,1))
        label = label[np.newaxis, ...]

        return img, label
