import numpy as np
import os.path
import argparse
import pdb
from pathlib import Path
import pickle
import cv2
import matplotlib.pyplot as plt
import pickle
import random


def make_parser():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--imgs_dir', type=str)
	parser.add_argument('--results_dir', type=str, default="tests_pca")
	parser.add_argument('--num_imgs', type=int, default=-1)
	return parser

def compute_pc_v2(img_paths, testing=False):

	num_imgs = len(img_paths)

	if num_imgs < 50:

		if testing:
			print ">> Naive algorithm"

		img = cv2.imread(img_paths[0])
		H, W = img.shape[:2]
		num_pixels = num_imgs * H * W
		pixels_values_total = img.reshape((-1, 3)).astype(float)
		#pixels_mean = np.zeros((1, 3), dtype=float)

		for img_path in img_paths[1:]:
			img = cv2.imread(img_path).astype(float)
			pixels_values = img.reshape((-1, 3))
			pixels_values_total = np.vstack((pixels_values_total, pixels_values))

		pixels_values_total /= 255.0
		cov_matrix = np.cov(pixels_values_total, rowvar=False)

	else:

		if testing:
			print ">> Two-pass algorithm"

		pixels_mean = np.zeros(3, dtype=float)

		for img_path in img_paths:
			img = cv2.imread(img_path).astype(float) / 255.0
			img_pixels = img.reshape((-1, 3))
			pixels_mean += np.mean(img_pixels, axis=0)

		pixels_mean /= num_imgs

		num_pixels = 0
		cov_matrix = np.zeros((3, 3), dtype=float)

		for img_path in img_paths:
			img = cv2.imread(img_path).astype(float) / 255.0
			img_pixels = img.reshape((-1, 3))
			img_pixels -= pixels_mean
			num_pixels += img_pixels.shape[0]
			cov_matrix += img_pixels.T.dot(img_pixels)

		cov_matrix /= float(num_pixels - 1)

	
	if testing:
		print ">> cov_matrix : {}".format(cov_matrix)
		#pdb.set_trace()


	eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
	pca = np.sqrt(eigen_vals) * eigen_vecs
	
	if testing:
		print ">> eigen_vals: {}".format(eigen_vals)
		print ">> eigen_vecs: {}".format(eigen_vecs)
		print ">> pca: {}".format(pca * 255)
		#pdb.set_trace()

	return (pca * 255.0)


def apply_pca_aug_v2(img, pca, sigma=0.5, testing=False):
	perturb = (pca * np.random.randn(3) * sigma).sum(axis=1)
	perturb = perturb[np.newaxis, np.newaxis, :]
	new_img = img + perturb
	if testing:
		print ">> perturb: {}".format(perturb[...,:])
		plt.figure()
		plt.imshow(img[...,::-1].astype(np.uint8))
		plt.figure()
		plt.imshow(new_img[...,::-1].astype(np.uint8))
		plt.show()
	return new_img


def pca_test_v2(results_dir, pca):

	test_imgs_dir = os.path.join(results_dir, "imgs")
	test_results_dir = os.path.join(results_dir, "results")

	for glob in Path(test_imgs_dir).glob("*.png"):
		img_name = glob.parts[-1]
		img_path = str(glob)
		new_img_path = os.path.join(test_results_dir, img_name)
		img = cv2.imread(img_path)
		new_img = apply_pca_aug_v2(img, pca, testing=True)
		cv2.imwrite(new_img_path, new_img)


if __name__ == "__main__":

	parser = make_parser()
	args = parser.parse_args()

	if args.imgs_dir is not None:

		img_paths = []
		for glob in Path(args.imgs_dir).glob("*.png"):
			img_paths.append(str(glob))

		random.shuffle(img_paths)
		if args.num_imgs < 0:
			pca = compute_pc_v2(img_paths, testing=True)
		else:
			pca = compute_pc_v2(img_paths[:args.num_imgs], testing=True)

		pca_test_v2(args.results_dir, pca)

