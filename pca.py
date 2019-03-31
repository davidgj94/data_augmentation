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
	return parser

def compute_pc_v2(img_paths, testing=False):

	num_imgs = len(img_paths)
	img = cv2.imread(img_paths[0])
	H, W = img.shape[:2]
	num_pixels = num_imgs * H * W
	pixels_values_total = img.reshape((-1, 3)).astype(float)
	#pixels_mean = np.zeros((1, 3), dtype=float)

	for img_path in img_paths[1:]:
		img = cv2.imread(img_path).astype(float)
		pixels_values = img.reshape((-1, 3))
		pixels_values_total = np.vstack((pixels_values_total, pixels_values))

	if testing:
		print ">> pixels_values_total shape : {}".format(pixels_values_total.shape)
		print ">> num_pixels: {}".format(num_pixels)
		#pdb.set_trace()

	pixels_values_total /= 255.0
	cov_matrix = np.cov(pixels_values_total, rowvar=False)

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

# def compute_pc(img_paths, testing=False):

# 	num_imgs = len(img_paths)
# 	pixels_mean = np.zeros((1, 3), dtype=float)

# 	for img_path in img_paths:
# 		img = cv2.imread(img_path).astype(float)
# 		img_pixels = img.reshape((-1, 3))
# 		pixels_mean += np.mean(img_pixels, axis=0)

# 	pixels_mean /= num_imgs

# 	num_pixels = 0
# 	cov_matrix = np.zeros((3, 3), dtype=float)

# 	for img_path in img_paths:
# 		img = cv2.imread(img_path).astype(float)
# 		img_pixels = img.reshape((-1, 3))
# 		#pdb.set_trace()
# 		img_pixels -= pixels_mean
# 		img_pixels /= 255.0
# 		num_pixels += img_pixels.shape[0]
# 		cov_matrix += img_pixels.T.dot(img_pixels)
# 		#pdb.set_trace()

# 	cov_matrix /= float(num_pixels - 1)

# 	# if testing:
# 	# 	print ">> cov_matrix : {}".format(cov_matrix)
# 	# 	print ">> pixels_mean: {}".format(pixels_mean)
# 	# 	print ">> pixels_mean: {}".format(num_pixels)
# 	# 	pdb.set_trace()

# 	eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
# 	pdb.set_trace()
# 	indices = np.argsort(eigen_vals)[::-1]
# 	eigen_vecs = eigen_vecs[:,indices[:3]]
# 	eigen_vals = eigen_vals[indices[:3]]
# 	#pdb.set_trace()

# 	if testing:
# 		print ">> eigen_vals: {}".format(eigen_vals)
# 		print ">> eigen_vecs: {}".format(eigen_vecs)
# 		pdb.set_trace()

# 	return pixels_mean, eigen_vecs, eigen_vals


# def apply_pca_aug(img, eigen_vecs, eigen_vals, mu=0, sigma=0.1):

# 	new_img = img.copy().astype(float)
# 	new_img /= 255.0

# 	scaled_eigen_vals = np.zeros(eigen_vals.shape, dtype=float)
# 	scaled_eigen_vals[0] = np.random.normal(mu, sigma) * eigen_vals[0]
# 	scaled_eigen_vals[1] = np.random.normal(mu, sigma) * eigen_vals[1]
# 	scaled_eigen_vals[2] = np.random.normal(mu, sigma) * eigen_vals[2]

# 	pca_noise = eigen_vecs.dot(scaled_eigen_vals)

# 	pdb.set_trace()
# 	print ">>> pca_noise: {}".format(pca_noise * 255.0)

# 	new_img[..., 0] += pca_noise[0]
# 	new_img[..., 1] += pca_noise[1]
# 	new_img[..., 2] += pca_noise[2]

# 	new_img *= 255.0

# 	return new_img

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


# def pca_test(results_dir, eigen_vecs, eigen_vals):

# 	test_imgs_dir = os.path.join(results_dir, "imgs")
# 	test_results_dir = os.path.join(results_dir, "results")

# 	for glob in Path(test_imgs_dir).glob("*.png"):
# 		img_name = glob.parts[-1]
# 		img_path = str(glob)
# 		new_img_path = os.path.join(test_results_dir, img_name)
# 		img = cv2.imread(img_path)
# 		new_img = apply_pca_aug(img, eigen_vecs, eigen_vals)
# 		cv2.imwrite(new_img_path, new_img)


if __name__ == "__main__":

	parser = make_parser()
	args = parser.parse_args()

	if args.imgs_dir is not None:

		img_paths = []
		for glob in Path(args.imgs_dir).glob("*.png"):
			img_paths.append(str(glob))

		random.shuffle(img_paths)
		pca = compute_pc_v2(img_paths[:50], testing=True)
		#pdb.set_trace()
		# pixels_mean, eigen_vecs, eigen_vals = compute_pc(img_paths, testing=True)

		# with open(os.path.join(args.results_dir, "parameters.pickle"), 'wb') as f:
		# 	pickle.dump((pixels_mean, eigen_vecs, eigen_vals), f)

		pca_test_v2(args.results_dir, pca)

	# else:

	# 	with open(os.path.join(args.results_dir, "parameters.pickle"), 'rb') as f:
	# 		pixels_mean, eigen_vecs, eigen_vals = pickle.load(f)

	# 	pca_test_v2(args.results_dir, eigen_vecs, eigen_vals)


