from utils import *
from image_augmentation import *
import numpy as np 
from tqdm import tqdm
from PIL import Image

from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.util import img_as_float, img_as_uint

def create_training_dataset(training_path = "..Data/InputData/Training Set/",	nrrd_MRI_file= "lgemri.nrrd", nrrd_label_file = 'laendo.nrrd', img_size = (576,576), nb_train = 1, crop = 'center', savedir = 'Dump'):

	print ('\n- Creating training dataset\n\n')

	create_folder(savedir)

	train_image_list, train_label_list = [], []

	training_files = os.listdir(training_path)

	if nb_train > len(training_files):
		print ('Number of test files required exceeding number of testing data available')
		print ('Adjusted to the maximum available:', len(training_files),'\n\n')
		nb_train = len(training_files)

	for file in tqdm(training_files[0:nb_train], ncols=100):

		train_image = load_nrrd(os.path.join(training_path, file, nrrd_MRI_file))
		train_label = load_nrrd(os.path.join(training_path, file, nrrd_label_file))//255 # Label values 0 or 1 (thanks to //255)

		###Center cropping
		if crop == 'center':
			train_image_crop = cropping(img_size, train_image)
			train_label_crop = cropping(img_size, train_label)

		for slice in range(train_label.shape[0]):
			
			###Center cropping
			if crop == 'center':
				train_image_list.append(train_image_crop[slice])
				train_label_list.append(train_label_crop[slice])

			###Centroid cropping
			if crop == 'centroid':

				if np.max(train_label[slice]) != 0:
					train_image_crop, train_label_crop = centroid_crop(img_size, train_image[slice], train_label[slice])

				else:
					train_image_crop = cropping(img_size, train_image[slice])
					train_label_crop = cropping(img_size, train_label[slice])

			train_image_list.append(train_image_crop)
			train_label_list.append(train_label_crop)

	train_image_list = np.array(train_image_list)
	train_label_list = np.array(train_label_list)

	return train_image_list, train_label_list


def create_testing_dataset(testing_path = "..Data/InputData/Testing Set/",	nrrd_MRI_file= "lgemri.nrrd", nrrd_label_file = 'laendo.nrrd', img_size = (576,576), nb_test = 10, crop = 'center',savedir = 'Dump'):

	print ('\n\n- Creating testing dataset\n\n')

	create_folder(savedir)

	testing_files = os.listdir(testing_path)

	if nb_test > len(testing_files):
		print ('Number of testing data requested exceeding number of testing data available')
		print ('Adjusted to the maximum available:', len(testing_files),'\n\n')
		nb_test = len(testing_files)

	for number in tqdm(range(nb_test), ncols=100):

		# Creating the folder for the testing data
		folder_name = savedir+"OutputData/"+str(number)+' - '+testing_files[number]
		create_folder (folder_name+'/Prediction')

		test_image_list, test_label_list = [], []

		test_image = load_nrrd(os.path.join(testing_path, testing_files[number], nrrd_MRI_file)) 
		test_label = load_nrrd(os.path.join(testing_path, testing_files[number], nrrd_label_file))//255 # Label values 0 or 1 (thanks to //255)

		if crop == 'center':
			test_image_crop = cropping(img_size, test_image)
			test_label_crop = cropping(img_size, test_label)


		for slice in range(test_label.shape[0]):

			if crop == 'center':
				test_image_list.append(test_image_crop[slice])
				test_label_list.append(test_label_crop[slice])

			###Centroid cropping
			if crop == 'centroid':
				if np.max(test_label[slice]) != 0:
					test_image_crop, test_label_crop = centroid_crop(img_size, test_image[slice], test_label[slice])
				else:
					test_image_crop = cropping(img_size, test_image[slice])
					test_label_crop = cropping(img_size, test_label[slice])

			# img = Image.fromarray(test_image[slice])
			# img.save(folder_name+'/Images/Slice%03d.tiff'%(slice))

			# label = Image.fromarray(test_label[slice]*255)
			# label.save(folder_name+'/Label/Slice%03d.tiff'%(slice))
			#test_image_clahe = equalize_adapthist(test_image[slice])

			test_image_list.append(test_image_crop)
			test_label_list.append(test_label_crop)

		test_image_list = np.array(test_image_list)
		test_label_list = np.array(test_label_list)

		np.save(folder_name+"/test_image.npy",test_image_list)
		np.save(folder_name+"/test_label.npy",test_label_list)


def image_norm(train_image_list):

	# calculate train mean and standard deviation
	train_mean = np.mean(train_image_list)
	train_SD = np.std(train_image_list)

	# Sample wise normalisation
	train_image_list = (train_image_list - train_mean)/train_SD

	#a = train_image_list.clip(min=0)

	return train_image_list, train_mean, train_SD


def data_augmentation(train_image_list, train_label_list, data_aug, probability = 0.5):

	print ('\n\n- Performing data augmentation...\n\n')

	train_image_aug, train_label_aug = [], []

	for slice in range(train_image_list.shape[0]):

		image_aug, label_aug = img_aug(train_image_list[slice], train_label_list[slice], data_aug, probability)

		train_image_aug.append(image_aug)
		train_label_aug.append(label_aug)

	train_image_aug = np.array(train_image_aug)
	train_label_aug = np.array(train_label_aug)

	return train_image_aug, train_label_aug


def data_reshape(train_image_list, train_label_list, img_size):

	# encoding label to neural network output format
	temp = np.empty(shape=[train_label_list.shape[0], img_size[1], img_size[0], 2])
	temp[:,:,:,0] = 1-train_label_list
	temp[:,:,:,1] = train_label_list

	train_image_list = np.reshape(train_image_list, newshape=[-1, img_size[1], img_size[0], 1])
	train_label_list = np.reshape(temp, newshape=[-1, img_size[1], img_size[0], 2])

	return train_image_list, train_label_list


if __name__ == '__main__':

	print ('Funky Funk')