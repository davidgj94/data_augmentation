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
	parser.add_argument('--results_dir', type=str, default="tests_illuminants/results")
	parser.add_argument('--illuminants_path', type=str)
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

if __name__ == "__main__":

	parser = make_parser()
	args = parser.parse_args()

	if args.illuminants_path is not None:

		illuminants = np.load(args.illuminants_path)
		indices = range(illuminants.shape[0])

		for glob in Path(args.imgs_dir).glob("*"):

			img_name = glob.parts[-1]
			img_path = os.path.join(args.imgs_dir, img_name)
			img = cv2.imread(img_path)
			white_img, _ = compute_illuminant(img, np.ones(img.shape, dtype=bool)[...,0])
			random.shuffle(indices)
			idx = indices[0]
			new_img = apply_illuminant(white_img, illuminants[idx,:].flatten())
			#pdb.set_trace()
			plt.figure()
			plt.imshow(img[...,::-1])
			plt.title("Original")
			plt.figure()
			plt.imshow(new_img[...,::-1].astype(np.uint8))
			plt.title("New")
			plt.show()


	else:

		illuminants_list = []
		img_list = []
		img_names_list = []

		for glob in Path(args.imgs_dir).glob("*"):

			img_name = glob.parts[-1]
			img_path = os.path.join(args.imgs_dir, img_name)

			img = cv2.imread(img_path)
			white_img, illuminant = compute_illuminant(img)

			illuminants_list.append(illuminant)
			img_list.append(white_img)
			img_names_list.append(img_name)

		
		for idx, img in enumerate(img_list):
			indices = range(0,len(img_list))
			indices.remove(idx)
			random.shuffle(indices)
			_idx = indices[0]
			new_img = apply_illuminant(img, illuminants_list[_idx])
			cv2.imwrite(os.path.join(args.results_dir, "{}__{}.png".format(img_names_list[idx], img_names_list[_idx])), new_img)