import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import os.path
import pdb
import matplotlib.pyplot as plt

def rotate_img(mat, angle, bbox_size, is_mask=False):

	height, width = mat.shape[:2]
	image_center = (width/2, height/2)

	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

	abs_cos = abs(rotation_mat[0,0])
	abs_sin = abs(rotation_mat[0,1])

	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)
	bound_size = (bound_w, bound_h)

	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]

	if is_mask:
		rotated_mat = cv2.warpAffine(mat, rotation_mat, bound_size, flags=cv2.INTER_NEAREST, borderValue=255)
	else:
		rotated_mat = cv2.warpAffine(mat, rotation_mat, bound_size, flags=cv2.INTER_AREA, borderValue=0)

	rotated_center = (bound_w/2, bound_h/2)

	x_limits = np.array((-bbox_size[0]/2, bbox_size[0]/2 + bbox_size[0]%2 - 1)) + rotated_center[0]
	y_limits = np.array((-bbox_size[1]/2, bbox_size[1]/2 + bbox_size[1]%2 - 1)) + rotated_center[1]

	bbox = rotated_mat[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]

	return bbox

# def rotate_img(mat, angle, bbox_size):
#   # angle in degrees

#   height, width = mat.shape[:2]
#   image_center = (width/2, height/2)

#   rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

#   abs_cos = abs(rotation_mat[0,0])
#   abs_sin = abs(rotation_mat[0,1])

#   bound_w = int(height * abs_sin + width * abs_cos)
#   bound_h = int(height * abs_cos + width * abs_sin)

#   rotation_mat[0, 2] += bound_w/2 - image_center[0]
#   rotation_mat[1, 2] += bound_h/2 - image_center[1]

#   bound_size = (bound_w, bound_h)
#   rotated_mat = cv2.warpAffine(mat, rotation_mat, bound_size, flags=cv2.INTER_AREA)
#   rotated_mask = cv2.warpAffine(np.zeros((height, width), dtype=np.uint8), rotation_mat, bound_size, flags=cv2.INTER_NEAREST, borderValue=255)

#   rotated_center = (bound_w/2, bound_h/2)

#   x_limits = np.array((-bbox_size[0]/2, bbox_size[0]/2 + bbox_size[0]%2 - 1)) + rotated_center[0]
#   y_limits = np.array((-bbox_size[1]/2, bbox_size[1]/2 + bbox_size[1]%2 - 1)) + rotated_center[1]

#   bbox = rotated_mat[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]
#   bbox_mask = rotated_mask[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]
  
#   return bbox, bbox_mask

def crop(img, top_border, bbox_size):

	x_limits = (top_border[0], top_border[0] + bbox_size[0])
	y_limits = (top_border[1], top_border[1] + bbox_size[1])

	bbox = img[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1]]

	return bbox

def random_crop(img, bbox_size):

	h, w = img.shape[:2]
	x_max = w - bbox_size[0]
	y_max = h - bbox_size[1]
	top_border_x = np.random.uniform(0, x_max, 1).astype(int)
	top_border_y = np.random.uniform(0, y_max, 1).astype(int)
	top_border = zip(top_border_x, top_border_y)[0]
	cropped_img = crop(img, top_border, bbox_size)

	return cropped_img, top_border


def make_parser():
    p = ArgumentParser()
    p.add_argument('--img_path', type=str, default="tests_transformations/img.png")
    p.add_argument('--save_path', type=str, default="tests_transformations/results")
    p.add_argument('--bbox_w', type=int, default=929)
    p.add_argument('--bbox_h', type=int, default=449)
    p.add_argument('-r', dest='do_rotation', action='store_true')
    p.set_defaults(do_rotation=False)
    p.add_argument('-f', dest='do_flip', action='store_true')
    p.set_defaults(do_flip=False)
    p.add_argument('-c', dest='do_crop', action='store_true')
    p.set_defaults(do_flip=False)
    return p

if __name__ == "__main__":

	parser = make_parser()
	args = parser.parse_args()

	img = cv2.imread(args.img_path)
	bbox_size = (args.bbox_w, args.bbox_h)

	if args.do_rotation:

		angles = np.linspace(-10,10,20)
		for angle in angles:
			bbox_img = rotate_img(img, angle, bbox_size)
			bbox_mask = rotate_img(np.zeros(tuple(img.shape[:2]), dtype=np.uint8), angle, bbox_size, is_mask=True)
			cv2.imwrite(os.path.join(args.save_path, "rotation", "imgs", "rotation_{}.png").format(int(angle)), bbox_img)
			cv2.imwrite(os.path.join(args.save_path, "rotation", "masks", "rotation_{}.png").format(int(angle)), bbox_mask)
	
	if args.do_crop:

		h, w = img.shape[:2]
		x_max = w - bbox_size[0]
		y_max = h - bbox_size[1]
		top_borders_x = np.random.uniform(0, x_max, 4).astype(int)
		top_borders_y = np.random.uniform(0, y_max, 4).astype(int)

		for top_border in zip(top_borders_x, top_borders_y):
			cropped_img = crop(img, top_border, bbox_size)
			cv2.imwrite(os.path.join(args.save_path, "crop", "{}_{}.png".format(top_border[0], top_border[1])), cropped_img)


	if args.do_flip:

		horizontal_img = cv2.flip( img, 0 )
		vertical_img = cv2.flip( img, 1 )
		both_img = cv2.flip( img, -1 )

		cv2.imwrite(os.path.join(args.save_path, "flip", "img.png"), img)
		cv2.imwrite(os.path.join(args.save_path, "flip", "horizontal_img.png"), horizontal_img)
		cv2.imwrite(os.path.join(args.save_path, "flip", "vertical_img.png"), vertical_img)
		cv2.imwrite(os.path.join(args.save_path, "flip", "both_img.png"), both_img)
