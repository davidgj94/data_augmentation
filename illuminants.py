import numpy as np
import os.path
import argparse
import pdb
from pathlib import Path
import pickle
import cv2
import matplotlib.pyplot as plt
import random

def apply_illuminant(img, illuminant):

	new_img = img.copy()
	new_img[..., 0] = new_img[..., 0] * illuminant[0] * np.sqrt(3)
	new_img[..., 1] = new_img[..., 1] * illuminant[1] * np.sqrt(3)
	new_img[..., 2] = new_img[..., 2] * illuminant[2] * np.sqrt(3)
	return new_img

def make_parser():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--imgs_dir', type=str, default="tests_illuminants/imgs")
	return parser

def compute_mink_norm(v, p=6):
	mink_norm = np.power(np.power(v, p).sum(), 1/float(p))
	return mink_norm

def compute_illuminant(img):

	B,G,R = np.dsplit(img, 3)

	B = np.squeeze(B).astype(float)
	G = np.squeeze(G).astype(float)
	R = np.squeeze(R).astype(float)

	white_B = compute_mink_norm(B.flatten())
	white_G = compute_mink_norm(G.flatten())
	white_R = compute_mink_norm(R.flatten())

	som = compute_mink_norm([white_B, white_G, white_R], p=2)

	white_B /= som
	white_G /= som
	white_R /= som

	new_B = B / (white_B * np.sqrt(3))
	new_G = G / (white_G * np.sqrt(3))
	new_R = R / (white_R * np.sqrt(3))

	white_img = np.dstack((new_B, new_G, new_R))

	return white_img, (white_B, white_G, white_R)

def test_illuminants(imgs):

	for i in range(len(imgs)):

		img1 = imgs[i]
		white_img, _ = compute_illuminant(img1)

		for j in range(len(imgs)):
			
			if i != j:

				img2 = imgs[j]
				_, illuminant = compute_illuminant(img2)

				new_img = apply_illuminant(white_img, illuminant)

				plt.figure()
				plt.imshow(img1[...,::-1])
				plt.title("Original Image")

				plt.figure()
				plt.imshow(new_img[...,::-1].astype(np.uint8))
				plt.title("New Image")

				plt.figure()
				plt.imshow(img2[...,::-1])
				plt.title("Illuminant Image")
				plt.show()


if __name__ == "__main__":

	parser = make_parser()
	args = parser.parse_args()

	imgs = []

	for glob in Path(args.imgs_dir).glob("*.png"):

		img = cv2.imread(str(glob))
		imgs.append(img)
		
	test_illuminants(imgs)
