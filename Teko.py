import os
from utils import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

if os.uname()[1] != "hpc2.bioeng.auckland.ac.nz":
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
	os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
from models import *
from data_process import *

from PIL import Image

def main():

	start_time = time.time()

	#################
	# Variable preparation
	#################

	# Variable set paths
	training_path = "../Data/InputData/_Training Set/"
	testing_path = "../Data/InputData/_Testing Set/"
	nrrd_MRI_file = "lgemri.nrrd"
	nrrd_LABEL_file = "laendo.nrrd"

	comments = "Teko - Unet Model " 

	# Variable set values
	session = "28"
	epoch = 300
	nb_train = 5
	nb_test = 5
	if training_path == "../Data/InputData/_Training Set/":
		nb_train = 100
		nb_test = 54
	img_size = (240, 240)
	batch_size = 44
	kernel_size = 5
	feature_map = 8
	lr = 0.001
	keep_rate = 0.8
	loss_function = "dice_loss"#"categorical_crossentropy"#"jacquard_2d"#
	weights = False#"Weights/baseline/Teko_5x5" #"baseline_Zhao/Zhao_5x5" #"baseline/Vnet_baseline_weights"


	data_aug = ["aug_rotation(image, label, (-25,25), probability = 1)","aug_scale(image, label, probability = 1)","aug_flipLR(image, label, probability = 1)", "aug_gammaContrast(image, label, probability = 1)"]#, "aug_shear(image, label, probability = 0.1)"]

	crop = 'centroid' #'center'/'centroid'

	''' Data augmentation available to put in the list of data augmentation
	"aug_rotation(image, label, (-20,20), probability = 1)" 
	"aug_scale(image, label, probability = 1)"
	"aug_perspective(image, label, probability = 1)"
	"aug_add(image, label, probability = 1)"
	"aug_contrastNorm(image, label, probability = 1)"
	"aug_gammaContrast(image, label, probability = 1)"
	"aug_elastic_T(image, labe, image.shape[0]*3, image.shape[0]*0.05)"
	"aug_unsharp(image, label, probability = 1)"
	'''
	# Directories creation
	savedir = "Results/Session_"+session+"/"
	mod_dir = savedir+'model/'
	log_dir = savedir+"logs"
	
	if not os.path.lexists(savedir):
		os.makedirs(savedir)
		os.makedirs(mod_dir)
		os.makedirs(log_dir)

	###########################################
	# Testing data processing and model loading
	###########################################

	print(ZNET())

	log(nrrd_LABEL_file,
		img_size,
		session,
		epoch,
		nb_train,
		nb_test,
		kernel_size,
		lr,
		loss_function, 
		batch_size, 
		weights,
		data_aug,
		crop,
		comments,
		savedir)


	### Loading model
	print ('\nLoading model...\n\n')
	# Building the model and loading the weights
	#model = Znet((None, img_size[1], img_size[0], 1), feature_map, kernel_size, keep_rate, lr, log_dir)
	#model = Inception((None, img_size[1], img_size[0], 1), feature_map, kernel_size, keep_rate, lr, log_dir)
	model = tf_Unet((None, img_size[1], img_size[0], 1),log_dir=log_dir)

	if weights:
		model.load(weights)


	################################
	# Datasets creation and Pre-processing
	################################

	#Creation of training dataset without data augmentation
	train_image_list, train_label_list = create_training_dataset(training_path, nrrd_MRI_file, nrrd_LABEL_file, img_size, nb_train, crop, savedir)

	#Creation of the testing dataset
	create_testing_dataset(testing_path, nrrd_MRI_file, nrrd_LABEL_file, img_size, nb_test, crop, savedir)

	#Dataset normalization
	train_image_list, train_mean, train_SD = image_norm(train_image_list)

	#Reshape to onehot
	train_image_list, train_label_list = data_reshape(train_image_list, train_label_list, img_size)


	################################
	# Training and Data augmentation
	################################

	best_dice = best_epoch = best_jacquard = 0

	score_list = np.zeros(1000)

	for e in range(epoch):

		tic = time.time()

		print ("\n\n\n --- Session", session, "- Epoch",str(e+1)+"/"+str(epoch)+'\n')

		Epoch_details = open(savedir+'Epoch_details.txt','a')
		Epoch_details.write("-"*75+" Epoch "+str(e+1)+"\n\n")
		Epoch_details.close()


		#Data augmentation and training
		if data_aug:
			
			train_image_aug, train_label_aug = data_augmentation(train_image_list, train_label_list, data_aug, probability = 0.5)
			model.fit(train_image_aug, train_label_aug, n_epoch=1, show_metric=True, batch_size=batch_size, shuffle=True)

		else :
			model.fit(train_image_list, train_label_list, n_epoch=1, show_metric=True, batch_size=batch_size, shuffle=True)


		###Prediction and evaluation
		dice_epoch, jacquard_epoch = pred(model, train_mean, train_SD, img_size, best_dice, savedir)
		score_list[e] = dice_epoch

		###Save the best results
		if dice_epoch >= best_dice:
			best_dice = dice_epoch
			best_epoch = e+1
			best_jacquard = jacquard_epoch

		###Saving the best model after epoch 20
		if e > 20 and dice_epoch >= best_dice:
			model.save(mod_dir+"model.tfl")

		r_change = open(savedir+'Rate_change.txt','a')
		r_change.write(str(rate_change(score_list, e))+"\n")
		r_change.close()


		###RAW scores 
		Raw_scores = open(savedir+'Raw_scores.txt','a')
		Raw_scores.write(str(dice_epoch)+'\n')
		Raw_scores.close()

		toc = time.time()

		print ('Remaining time:', time.strftime("%H:%M:%S", time.gmtime((toc-tic)*(epoch-(e+1)))), '\n')

	###Final details

	msd_mean, hd_mean, diam_err, vol_err = metrics(savedir)

	elapsed_time = time.time() - start_time

	Session_log = open(savedir+'Details.txt','a')
	Session_log.write("\n"+"-"*30+'\n')
	Session_log.write("Best Epoch: "+str(best_epoch)+"\n")
	Session_log.write("\nDice score: "+str(best_dice))
	Session_log.write("\nJacquard score: "+str(best_jacquard))
	Session_log.write("\nMean Surface Distance: "+str(msd_mean)+"mm")
	Session_log.write("\nHausdorff Distance: "+str(hd_mean)+"mm")
	Session_log.write("\nDiameter error: "+str(diam_err*100))
	Session_log.write("\nVolume error: "+str(vol_err*100))
	Session_log.write('\n\nTotal time : ' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + '\n')
	Session_log.close()

	total = open('Results/Resume.txt','a')
	total.write(savedir+": "+str(best_dice)+'\n')
	total.close()

	plot_learn(savedir)

	resume_display(session, best_epoch, best_dice, best_jacquard, elapsed_time)

if __name__ == '__main__':

	main()
