# -*- coding: utf-8 -*-
import os, glob, re, random, time, datetime, resource
import numpy as np
import SimpleITK as sitk
import cv2
from scipy import ndimage
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

def atoi(text):
	return int(text) if text.isdigit() else text

def limit_memory(maxsize):
	soft, hard = resource.getrlimit(resource.RLIMIT_AS)
	resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def natural_keys(text):
	''' alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)'''
	return [ atoi(c) for c in re.split('(\d+)', text) ]


def glab(path):
	'''Glob with natural sorting'''
	return sorted(glob.glob(path),key=natural_keys)


def create_folder(full_path_filename):
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)

	return


def load_nrrd(full_path_filename):
	'''this function loads .nrrd files into a 3D matrix and outputs it
	the input is the specified file path to the .nrrd file'''
	data = sitk.ReadImage( full_path_filename )
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
	data = sitk.GetArrayFromImage(data)

	return data



def log(img_type, 
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
	savedir):

	'''Writing the log file containing important information
	to track the changes and parameters'''

	file = open(savedir+'Details.txt','w+')

	date = datetime.datetime.now()

	file.write(RTBF())

	file.write("Date : "+date.strftime("%d/%m/%Y, %H:%M:%S")+'\n')
	file.write("Machine:"+os.uname()[1]+'\n')
	file.write("-"*30+'\n')
	file.write("Saving directory : "+savedir+'\n')
	file.write("Images size : "+str(img_size[0])+'x'+str(img_size[1])+'\n')
	file.write("Images processed : "+str(img_type)+'\n')
	file.write("Number of Epoch = "+str(epoch)+'\n')
	file.write("Number of Training data = "+str(nb_train)+'\n')
	file.write("Number of Testing data = "+str(nb_test)+'\n')
	file.write("Filters size = "+str(kernel_size)+'\n')
	file.write("Learning rate = "+str(lr)+'\n')
	file.write("Loss function = "+str(loss_function)+'\n')
	file.write("Batch size = "+str(batch_size)+'\n')
	file.write("Weights loaded = "+str(weights)+'\n')
	file.write("Data augmentation = "+str(data_aug)+'\n\n')
	file.write("Cropping method = "+str(crop)+'\n\n')
	file.write("Comments :"+str(comments)+"\n")

	file.close()


	#Displaying the different components of the run
	print ('*'*15,"Session",session,'*'*15,"\n")
	print (" - Date :", date.strftime("%d/%m/%Y, %H:%M:%S"))
	print (" - Saving directory :", savedir)
	print (" - Images size :", str(img_size[0]), 'x', str(img_size[1]))
	print (" - Number of Epoch :", str(epoch))
	print (" - Number of Training data :", str(nb_train))
	print (" - Number of Testing data :", str(nb_test))
	print (" - Filters size :", str(kernel_size))
	print (" - Learning rate :", str(lr))
	print (" - Loss function :", str(loss_function))
	print (" - Batch size :", str(batch_size))
	print (" - Weights loaded :", str(weights))
	print (" - Data augmentation :", str(data_aug))
	print (" - Cropping method : "+str(crop), '\n')
	print (" - Comments :", str(comments),"\n")
	print ("*"*41,"\n")


def resume_display(session, best_epoch, best_dice, best_jacquard, elapsed_time):
	print ('\n\n')
	print ('*'*15,'Best score','*'*15)
	print ('*'.ljust(40),'*')
	print ('* \t\tSession :',session,' '.ljust(12),'*')
	print ('*'.ljust(40),'*')
	print ('* \tBest epoch:\t', best_epoch,' '.ljust(10),'*')
	print ('* \tDice score:\t', np.round(best_dice,4),' '.ljust(8),'*')
	print ('* \tJacquard score:\t', np.round(best_jacquard,4),' '.ljust(8),'*')
	print ('*'.ljust(40),'*')
	print ('* \tTotal time :', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),' '.ljust(10),'*')
	print ('*'.ljust(40),'*')
	print ('*'*42,'\n')
	

def cropping(img_size, data):

	# Base on the image size cropped image according to the desired size
	if len(data.shape) == 4:
		midpoint = data.shape[2]//2	
	else:
		midpoint = data.shape[1]//2	

	startx, endx = midpoint - int(img_size[1]/2), midpoint + int(img_size[1]/2)
	starty, endy = midpoint - int(img_size[0]/2), midpoint + int(img_size[0]/2)

	if len(data.shape) == 2:
		data = data[startx:endx, starty:endy]
	elif len(data.shape) == 3:
		data = data[0:data.shape[0], startx:endx, starty:endy]
	elif len(data.shape) == 4:
		data = data[0:data.shape[0], 0:data.shape[1], startx:endx, starty:endy]
	else:
		print ("\nERROR: Bad cropping shape")
		quit()

	return data


def centroid_crop(img_size, image, label):

	centroid = ndimage.measurements.center_of_mass(label)
	y, x, half_crop = np.int(np.round(centroid[0])), np.int(np.round(centroid[1])), (np.int(img_size[1]/2),np.int(img_size[0]/2))

	##When the centroid is no find this prevent the crash (centroid values <0)
	#using normal cropping
	if (y-half_crop[0]) < 0 or (x-half_crop[1]) < 0:
		
		return  cropping(img_size, image), cropping(img_size, label)

	if len(label.shape) == 3:
		label = label[0:label.shape[0], y-half_crop[0]:y+half_crop[0], x-half_crop[1]:x+half_crop[1]]
		image = image[0:image.shape[0], y-half_crop[0]:y+half_crop[0], x-half_crop[1]:x+half_crop[1]]
	elif len(label.shape) == 2:
		label = label[y-half_crop[0]:y+half_crop[0], x-half_crop[1]:x+half_crop[1]]
		image = image[y-half_crop[0]:y+half_crop[0], x-half_crop[1]:x+half_crop[1]]
	else:
		print ("\nERROR: Bad cropping shape")
		quit()
	
	if label.shape != (img_size[0], img_size[1]):
		label = np.pad(label, ((0, img_size[1]-label.shape[0]), (0, img_size[0]-label.shape[1])), 'reflect')
		image = np.pad(image, ((0, img_size[1]-image.shape[0]), (0, img_size[0]-image.shape[1])), 'reflect')

	return image, label


def file_len(fname):
    return sum(1 for line in open(fname))


def plot_learn(savedir):

	file_path = savedir+"Raw_scores.txt"

	num_lines = file_len(file_path)

	## Collecting the scores
	with open(file_path, "r") as file:
	
		scores = []
		epochs = np.arange(1, num_lines+1)
		for number, line in enumerate(file):
			scores.append(float(line.rstrip()))

	## Finding max position
	ymax = max(scores)
	xmax = epochs[scores.index(ymax)]

	## Creating the figure
	fig = plt.figure()
	ax = fig.add_subplot(111)

	## Generating the graph
	line, = ax.plot(epochs, scores, linewidth=0.8)

	## Setting the limit
	ax.set_ylim(min(scores), 0.97)

	## Marking max position with X
	ax.plot(xmax, ymax, "x", ms=10, markerfacecolor="None",
         markeredgecolor='red', markeredgewidth=0.8)
	## With the Dice score value
	ymax = round(ymax,4)
	ax.annotate(ymax, xy=(xmax, ymax), xytext=(xmax-5.5, ymax+0.005))

	## Title and axis label
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Dice score')
	ax.set_title('Training dice score evolution -'+savedir)

	## Saving the images (PDF for final figures)
	plt.savefig(savedir+'Training.png', bbox_inches='tight')
	#plt.savefig(savedir+')Training.pdf', bbox_inches='tight')
	#plt.show()


	def getLargestCC(segmentation):
		
		labels = label(segmentation)
		unique, counts = np.unique(labels, return_counts=True)
		list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
		largest=max(list_seg, key=lambda x:x[1])[0]
		labels_max=(labels == largest).astype(int)

		return labels_max


def pred(CNN_model, mu = 0.2, sd = 0.1, img_size = (576,576), best_score = 0, savedir = 'Dump'):

	test_folders = glab(savedir+"OutputData/*")

	details = open(savedir+"Epoch_details.txt","a")
	
	print ("\nPredicting ...\n\n")

	predict_dict = {}
	groundt_dict = {}

	dice_scores = []
	jacquard_scores = []
	centroid_mse = (0,0)

	k = 1 #Value for the class evaluated

	for folder in tqdm(test_folders, ncols=100):
		groundT = []
		test_image = np.load(folder+'/test_image.npy')
		ground_truth = np.load(folder+'/test_label.npy')

		# Numpy allocation arrays
		prediction = np.zeros(shape=[test_image.shape[0], img_size[1], img_size[0]])
		temp_Input = np.zeros(shape=[test_image.shape[0], img_size[1], img_size[0]])
		temp_Output = np.zeros(shape=[test_image.shape[0], img_size[1], img_size[0], 2])

		for number, slice in enumerate(test_image):

			temp_Input[number,:,:] = (slice - mu)/sd
			temp_Output[number,:,:,:] = CNN_model.predict([temp_Input[number,:,:,None]])
			groundT.append(ground_truth[number])

		prediction = np.argmax(temp_Output, 3)
		groundT = np.array(groundT)

		#Dice Score (f1 score)
		dice = np.sum(prediction[groundT==k]==k)*2.0 / (np.sum(prediction[prediction==k]==k) + np.sum(groundT[groundT==k]==k))
		#Jacquard Index (Intersection over Union)
		IoU = np.sum(prediction[groundT==k]==k) / (np.sum(prediction[prediction==k]==k) + np.sum(groundT[groundT==k]==k) - (np.sum(prediction[groundT==k]==k)))

		dice_scores.append(dice)
		jacquard_scores.append(IoU)

		dice_average = np.mean(np.array(dice_scores))
		jacquard_average = np.mean(np.array(jacquard_scores))

		details.write(os.path.basename(folder)+" Dice score: "+str(dice)+"\n")

		# Prediction save
		predict_dict[folder] = np.array(prediction)
		groundt_dict[folder] = np.array(groundT)	


	print ('\n\nDice score : ', dice_average)
	print ('\nJacquard score : ', jacquard_average)
	details.write("\nOverall Dice Average = "+str(dice_average))
	details.write("\nOverall Jacquard Average = "+str(jacquard_average)+"\n\n")
	details.close()


	if dice_average > best_score:

		print ('\n\nSaving prediction...\n\n')
		for folder in predict_dict:

			np.save(folder+'/prediction.npy', predict_dict[folder])
			for number, slice in enumerate(predict_dict[folder]):

				#Saving the predicted images
				cv2.imwrite(folder+"/Prediction/Slice%03d.tiff"%(number), 255 * slice)

	return dice_average, jacquard_average


def metrics(savedir):

	folder = glab(savedir+"OutputData/*")

	diam_error = np.zeros([len(folder)])
	Vol_error = np.zeros([len(folder)])

	hd = np.zeros([len(folder)])
	msd = np.zeros([len(folder)])

	for index, value in enumerate(folder):

		prediction = np.load(value+"/prediction.npy")
		ground_truth = np.load(value+"/test_label.npy")

		surface_distance = surfd(prediction, ground_truth, [0.625, 0.625, 0.625], 1)

		msd[index] = surface_distance.mean()
		hd[index] = surface_distance.max()

		diam_error[index], Vol_error[index] = diam_vol_err(prediction, ground_truth)

	return np.mean(msd), np.mean(hd), np.mean(diam_error), np.mean(Vol_error)


def surfd(input1, input2, sampling=1, connectivity=1):
    """Symmetric surface distance calculation
	OP: https://mlnotebook.github.io/post/surface-distance-function/

    Usage : 1] surface_distance = surfd(test_seg, GT_seg, [1.25, 1.25, 10],1)
        	2] surface_distance = surfd(test_seg(test_seg==1), \
                   GT_seg(GT_seg==1), [1.25, 1.25, 10],1)
    Calculations: 
    msd = surface_distance.mean() (Mean Surface Distance)
    rms = np.sqrt((surface_distance**2).mean()) (Residual Mean Square Distance) More sensitive for smaller values (used for comparison with close model. To be used 'as' jacquard)
    hd  = surface_distance.max() (Hausdorff distance)
    """

    # Check of binary images, all values >1 == 1 (True)
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)
    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])       
    
    return sds


def diam_vol_error(prediction, ground_truth):

	folder = glab(savedir+"OutputData/*")

	diam_error = np.zeros([len(folder)])
	Vol_error = np.zeros([len(folder)])

	for number, name in enumerate(folder):

		prediction = np.load(name+"/prediction.npy")
		groundT = np.load(name+"/test_label.npy")

		diam_pred = np.max(np.sum(prediction, axis=2))
		diam_GT = np.max(np.sum(groundT, axis=2))

		diam_error[number] = np.abs(diam_pred-diam_GT)/diam_GT
		Vol_error[number] = np.abs(np.sum(prediction==1)-np.sum(groundT==1))/np.sum(groundT==1)

		#diam_error.append(np.abs(diam_pred-diam_GT)/diam_GT)
		#Vol_error.append(np.abs(np.sum(prediction==1)-np.sum(groundT==1))/np.sum(groundT==1))

	return np.mean(diam_error), np.mean(Vol_error)


def diam_vol_err(prediction, ground_truth):

	diam_pred = np.max(np.sum(prediction, axis=2))
	diam_GT = np.max(np.sum(ground_truth, axis=2))

	diam_error = np.abs(diam_pred-diam_GT)/diam_GT
	Vol_error = np.abs(np.sum(prediction==1)-np.sum(ground_truth==1))/np.sum(ground_truth==1)

	return diam_error, Vol_error


def rate_change(scores, epoch):
	#Calculation of the rate of change
	#on rolling window for 40 epochs
	rate = np.zeros(39)

	for i in range(epoch-39, epoch):

		rate[i-epoch] = scores[i+1] - scores[i]

	return np.sum(rate)



def RTBF():
	return ('''

  _____       _        _ _     
 |  __ \     | |      (_) |    
 | |  | | ___| |_ __ _ _| |___ 
 | |  | |/ _ \ __/ _` | | / __|
 | |__| |  __/ || (_| | | \__ \

 |_____/ \___|\__\__,_|_|_|___/
                               
                               
''')


def ZNET():

	return ('''

████████╗███████╗██╗  ██╗ ██████╗ 
╚══██╔══╝██╔════╝██║ ██╔╝██╔═══██╗
   ██║   █████╗  █████╔╝ ██║   ██║
   ██║   ██╔══╝  ██╔═██╗ ██║   ██║
   ██║   ███████╗██║  ██╗╚██████╔╝
   ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝ 
                                                                     
        ''')
if __name__ == '__main__':

	print ('Banana song')

	img = cv2.imread('../Data/FullDataset/Training/3 - 2XL5HSFSE93RMOJDRGR4/Images/Slice045.tiff',0)
	print (img.shape)

	lbl = np.array(Image.open('../Data/FullDataset/Training/3 - 2XL5HSFSE93RMOJDRGR4/Label/Slice045.tiff'))

	stack = np.zeros([2,576,576])
	stack[0] = img
	stack[1] = lbl

	#stack2 = np.zeros([4,576,576])
	stack2 = np.repeat(stack,2, axis=0)
	print(stack2.shape)
	img = Image.fromarray(stack2[0])
	img.show()
	time.sleep(1)
	img = Image.fromarray(stack2[1])
	img.show()
	time.sleep(1)
	img = Image.fromarray(stack2[2])
	img.show()
	time.sleep(1)
	img = Image.fromarray(stack2[3])
	img.show()
	# img_size = (240,160)

	# image, label = centroid_crop(img_size, img, lbl)

	# image2 = cropping(img_size, img)

	# print (image.shape)
	# print (label.shape)
	# img = Image.fromarray(image)
	# img.show()

	# img2 = Image.fromarray(image2)
	# img2.show()
