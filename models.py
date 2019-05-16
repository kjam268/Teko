from __future__ import division, print_function, absolute_import
from utils import create_folder
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data ,dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_2d_transpose, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN



def Znet(input_size = (80, 576, 576, 2), feature_map=8, kernel_size=5, keep_rate=0.8, lr=0.001, log_dir ="logs"):

	# 2d convolution operation
	def tflearn_conv_2d(net, nb_filter, kernel, stride,dropout=1.0,activation=True):

		net = tflearn.layers.conv.conv_2d(net, nb_filter, kernel, stride, padding="same", activation="linear",bias=False)
		net = tflearn.layers.normalization.batch_normalization(net)
		
		if activation:
			#net = tflearn.activations.prelu(net)
			net = tflearn.activations.leaky_relu(net)
		
		net = tflearn.layers.core.dropout(net, keep_prob=dropout)
		
		return(net)

	# 2d deconvolution operation
	def tflearn_deconv_2d(net, nb_filter, kernel, stride, dropout=1.0):

		net = tflearn.layers.conv.conv_2d_transpose(net, nb_filter, kernel,
													[net.shape[1].value*stride, net.shape[2].value*stride, nb_filter],
													[1, stride, stride,1],padding="same",activation="linear",bias=False)
		net = tflearn.layers.normalization.batch_normalization(net)
		#net = tflearn.activations.prelu(net)
		net = tflearn.activations.leaky_relu(net)
		net = tflearn.layers.core.dropout(net, keep_prob=dropout)
		
		return(net)

	# merging operation
	def tflearn_merge_2d(layers, method):
		
		net = tflearn.layers.merge_ops.merge(layers, method, axis=3)
		
		return(net)



	# level 0 input
	layer_0a_input	= tflearn.layers.core.input_data(input_size) #shape=[None,n1,n2,n3,1])
	print ('input',layer_0a_input.shape)


	print ('*'*20,'Block 1','*'*20)
	# level 1 down
	layer_1a_conv 	= tflearn_conv_2d(net=layer_0a_input, nb_filter=feature_map, kernel=5, stride=1,activation=False)
	print ('1a_conv',layer_1a_conv.shape)

	layer_1a_stack	= tflearn_merge_2d([layer_0a_input]*feature_map, "concat")
	print ('1a_concat',layer_1a_stack.shape)
	layer_1a_stack 	= tflearn.activations.prelu(layer_1a_stack)

	layer_1a_add	= tflearn_merge_2d([layer_1a_conv,layer_1a_stack], "elemwise_sum")
	print ('1a_add',layer_1a_add.shape)

	layer_1a_down	= tflearn_conv_2d(net=layer_1a_add, nb_filter=feature_map*2, kernel=2, stride=2)
	print ('1a_down',layer_1a_down.shape)


	print ('*'*20,'Block 2','*'*20)
	# level 2 down
	layer_2a_conv 	= tflearn_conv_2d(net=layer_1a_down, nb_filter=feature_map*2, kernel=kernel_size, stride=1)
	print ('2a_conv',layer_2a_conv.shape)

	layer_2a_conv 	= tflearn_conv_2d(net=layer_2a_conv, nb_filter=feature_map*2, kernel=kernel_size, stride=1)
	print ('2.1a_conv',layer_2a_conv.shape)

	layer_2a_add	= tflearn_merge_2d([layer_1a_down,layer_2a_conv], "elemwise_sum")
	print ('2a_add',layer_2a_add.shape)

	layer_2a_down	= tflearn_conv_2d(net=layer_2a_add, nb_filter=feature_map*4, kernel=2, stride=2)
	print ('2a_down',layer_2a_down.shape)


	print ('*'*20,'Block 3','*'*20)
	# level 3 down
	layer_3a_conv 	= tflearn_conv_2d(net=layer_2a_down, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	print ('3a_conv',layer_3a_conv.shape)

	layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	print ('3.1a_conv',layer_3a_conv.shape)

	layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	print ('3.2a_conv',layer_3a_conv.shape)

	layer_3a_add	= tflearn_merge_2d([layer_2a_down,layer_3a_conv], "elemwise_sum")
	print ('3a_add',layer_3a_add.shape)

	layer_3a_down	= tflearn_conv_2d(net=layer_3a_add, nb_filter=feature_map*8, kernel=2, stride=2, dropout=keep_rate)
	print ('3a_down',layer_3a_down.shape)	


	print ('*'*20,'Block 4','*'*20)
	# level 4 down
	layer_4a_conv 	= tflearn_conv_2d(net=layer_3a_down, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('4a_conv',layer_4a_conv.shape)

	layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('4.1a_conv',layer_4a_conv.shape)

	layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('4.2a_conv',layer_4a_conv.shape)

	layer_4a_add	= tflearn_merge_2d([layer_3a_down,layer_4a_conv], "elemwise_sum")
	print ('4a_add',layer_4a_add.shape)

	layer_4a_down	= tflearn_conv_2d(net=layer_4a_add, nb_filter=feature_map*16,kernel=2,stride=2,dropout=keep_rate)
	print ('4a_down',layer_4a_down.shape)


	print ('*'*20,'Block 5','*'*20)
	# level 5
	layer_5a_conv 	= tflearn_conv_2d(net=layer_4a_down, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('5a_conv',layer_5a_conv.shape)	

	layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('5.1a_conv',layer_5a_conv.shape)	

	layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('5.2a_conv',layer_5a_conv.shape)

	layer_5a_add	= tflearn_merge_2d([layer_4a_down,layer_5a_conv], "elemwise_sum")
	print ('5a_add',layer_5a_add.shape)

	layer_5a_up		= tflearn_deconv_2d(net=layer_5a_add, nb_filter=feature_map*8, kernel=2, stride=2, dropout=keep_rate)
	print ('5a_up',layer_5a_up.shape)


	print ('*'*20,'Block 4 up','*'*20)
	# level 4 up
	layer_4b_concat	= tflearn_merge_2d([layer_4a_add,layer_5a_up], "concat")
	print ('4b_concat',layer_4b_concat.shape)

	layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_concat, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('4b_conv',layer_4b_conv.shape)

	layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('4.1b_conv',layer_4b_conv.shape)

	layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('4.2b_conv',layer_4b_conv.shape)

	layer_4b_add	= tflearn_merge_2d([layer_4b_conv,layer_4b_concat], "elemwise_sum")
	print ('4b_add',layer_4b_add.shape)

	layer_4b_up		= tflearn_deconv_2d(net=layer_4b_add, nb_filter=feature_map*4, kernel=2, stride=2, dropout=keep_rate)
	print ('4b_up',layer_4b_up.shape)


	print ('*'*20,'Block 3 up','*'*20)	
	# level 3 up
	layer_3b_concat	= tflearn_merge_2d([layer_3a_add,layer_4b_up], "concat")
	print ('3b_concat',layer_3b_concat.shape)

	layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_concat, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('3b_conv',layer_3b_conv.shape)

	layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('3.1b_conv',layer_3b_conv.shape)

	layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	print ('3.2b_conv',layer_3b_conv.shape)

	layer_3b_add	= tflearn_merge_2d([layer_3b_conv,layer_3b_concat], "elemwise_sum")
	print ('3b_add',layer_3b_add.shape)

	layer_3b_up		= tflearn_deconv_2d(net=layer_3b_add, nb_filter=feature_map*2, kernel=2, stride=2)
	print ('3b_up',layer_3b_up.shape)


	print ('*'*20,'Block 2 up','*'*20)
	# level 2 up
	layer_2b_concat	= tflearn_merge_2d([layer_2a_add,layer_3b_up], "concat")
	print ('2b_concat',layer_2b_concat.shape)

	layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_concat, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	print ('2b_conv',layer_2b_conv.shape)

	layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_conv, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	print ('2.1b_conv',layer_2b_conv.shape)

	layer_2b_add	= tflearn_merge_2d([layer_2b_conv,layer_2b_concat], "elemwise_sum")
	print ('2b_add',layer_2b_add.shape)

	layer_2b_up		= tflearn_deconv_2d(net=layer_2b_add, nb_filter=feature_map, kernel=2, stride=2)
	print ('2b_up',layer_2b_up.shape)


	print ('*'*20,'Block 1 up','*'*20)
	# level 1 up
	layer_1b_concat	= tflearn_merge_2d([layer_1a_add,layer_2b_up], "concat")
	print ('1b_concat',layer_1b_concat.shape)

	layer_1b_conv 	= tflearn_conv_2d(net=layer_1b_concat, nb_filter=feature_map*2, kernel=kernel_size, stride=1)
	print ('1b_conv',layer_1b_conv.shape)

	layer_1b_add	= tflearn_merge_2d([layer_1b_conv,layer_1b_concat], "elemwise_sum")
	print ('1b_add',layer_1b_add.shape)


	print ('*'*20,'Block 0 classifier','*'*20)	
	# level 0 classifier
	layer_0b_conv	= tflearn_conv_2d(net=layer_1b_add, nb_filter=2, kernel=5, stride=1)
	print ('0b_conv',layer_0b_conv.shape)

	layer_0b_clf	= tflearn.layers.conv.conv_2d(layer_0b_conv, 2, 1, 1, activation="softmax")
	print ('0b_classifier',layer_0b_clf.shape)

	# Optimizer
	regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam', loss=dice_loss_2d, learning_rate=lr) # categorical_crossentropy/dice_loss_3d

	model   = tflearn.models.dnn.DNN(regress, tensorboard_dir=log_dir)

	# Saving the model
	create_folder("Weights/baseline/")
	model.save("Weights/baseline/Teko_5x5")

	return model


def tf_Unet(input_size = (576, 576, 2), kernel_size=3, lr=0.0001, loss_function="categorical_crossentropy"):

	block1a = input_data(input_size)#shape=[None,320,320,1])
	block1a = conv_2d(block1a, 64, kernel_size, activation='relu')
	block1a = conv_2d(block1a, 64, kernel_size, activation='relu')

	#block2a = conv_2d(block1a, 2, 2, 2, activation='relu')
	block2a = max_pool_2d(block1a,  2, 2)
	block2a = conv_2d(block2a, 128, kernel_size, activation='relu')
	block2a = conv_2d(block2a, 128, kernel_size, activation='relu')


	#block3a = conv_2d(block2a, 2, 2, 2, activation='relu')
	block3a = max_pool_2d(block2a,  2, 2)
	block3a = conv_2d(block3a, 256, kernel_size, activation='relu')
	block3a = conv_2d(block3a, 256, kernel_size, activation='relu')

	#block4a = conv_2d(block3a, 2, 2, 2, activation='relu')
	block4a = max_pool_2d(block3a,  2, 2)
	block4a = conv_2d(block4a, 512, kernel_size, activation='relu')
	block4a = conv_2d(block4a, 512, kernel_size, activation='relu')
	block4a = dropout(block4a, 0.75)

	#block5 = conv_2d(block4a, 2, 2, 2, activation='relu')
	block5 = max_pool_2d(block4a,  2, 2)
	block5 = conv_2d(block5, 1024, kernel_size, activation='relu')
	block5 = conv_2d(block5, 1024, kernel_size, activation='relu')
	block5 = dropout(block5, 0.75) #25% dropout [here is the keep rate]

	block4b = conv_2d_transpose(block5, 512, kernel_size, [block5.shape[1].value*2, block5.shape[2].value*2, 512], [1,2,2,1])
	block4b = merge([block4a, block4b], 'concat', axis=3)
	block4b = conv_2d(block4b, 512, kernel_size, activation='relu')
	block4b = conv_2d(block4b, 512, kernel_size, activation='relu')

	block3b = conv_2d_transpose(block4b, 256, kernel_size, [block4b.shape[1].value*2, block4b.shape[2].value*2, 256], [1,2,2,1])
	block3b = merge([block3a, block3b], 'concat', axis=3)
	block3b = conv_2d(block3b, 256, kernel_size, activation='relu')
	block3b = conv_2d(block3b, 256, kernel_size, activation='relu')

	block2b = conv_2d_transpose(block3b, 128, kernel_size, [block3b.shape[1].value*2, block3b.shape[2].value*2, 128], [1,2,2,1])
	block2b = merge([block2a, block2b], 'concat', axis=3)
	block2b = conv_2d(block2b, 128, kernel_size, activation='relu')
	block2b = conv_2d(block2b, 128, kernel_size, activation='relu')

	block1b = conv_2d_transpose(block2b, 64, kernel_size, [block2b.shape[1].value*2, block2b.shape[2].value*2, 64], [1,2,2,1])
	block1b = merge([block1a, block1b], 'concat', axis=3)
	block1b = conv_2d(block1b, 64, kernel_size, activation='relu')
	block1b = conv_2d(block1b, 64, kernel_size, activation='relu')

	Clf     = conv_2d(block1b, 2, 1, 1, activation='softmax')
	regress = regression(Clf, optimizer='adam', loss=dice_loss_2d, learning_rate=lr)
	model   = DNN(regress, tensorboard_verbose=0)

	### First initial model saved 
	#model.save("Weights/baseline_5x5/baseline_weights_5x5")

	return model
	

def dice_loss_2d(y_pred, y_true):
	
	with tf.name_scope("dice_loss_2D_function"):
		
		# foreground
		y_pred = y_pred[:,:,:,1]
		y_true = y_true[:,:,:,1]

		smooth = 1.0
		
		intersection = tf.reduce_sum(y_pred*y_true)
		union = tf.reduce_sum(y_pred*y_pred) + tf.reduce_sum(y_true*y_true)
		
		dice = (2.0 * intersection + smooth) / (union + smooth)
		
	return(1 - dice)


def jacquard_2d(y_pred, y_true):
	
	with tf.name_scope("jacquard_2D_function"):
		
		# foreground
		y_pred = y_pred[:,:,:,1]
		y_true = y_true[:,:,:,1]

		smooth = 1.0
		
		intersection = tf.reduce_sum(y_pred*y_true)
		union = tf.reduce_sum(y_pred*y_pred) + tf.reduce_sum(y_true*y_true)
		
		jacquard = (intersection + smooth) / (union + smooth)
		
	return(1 - jacquard)


def Inception(input_size = (80, 576, 576, 2), feature_map=8, kernel_size=5, keep_rate=0.8, lr=0.001, log_dir ="logs"):

	# 2d deconvolution operation
	def tflearn_deconv_2d(net, nb_filter, kernel, stride, dropout=1.0):

		net = tflearn.layers.conv.conv_2d_transpose(net, nb_filter, kernel,
													[net.shape[1].value*stride, net.shape[2].value*stride, nb_filter],
													[1, stride, stride,1],padding="same",activation="linear",bias=False)
		net = tflearn.layers.normalization.batch_normalization(net)
		#net = tflearn.activations.prelu(net)
		net = tflearn.activations.leaky_relu(net)
		net = tflearn.layers.core.dropout(net, keep_prob=dropout)
		
		return(net)

	network = input_data(input_size)
	conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
	pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
	pool1_3_3 = local_response_normalization(pool1_3_3)
	conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
	conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
	conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

	# 3a
	inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
	inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3,  activation='relu', name='inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
	inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
	inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	# 3b
	inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
	inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name='inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name='inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu', name='inception_3b_pool_1_1')
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')
	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

	# 4a
	inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
	inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
	inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

	# 4b
	inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
	inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
	inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
	inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
	inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
	inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
	inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

	# 4c
	inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
	inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
	inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
	inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
	inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
	inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
	inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
	inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3, name='inception_4c_output')

	# 4d
	inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
	inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
	inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
	inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
	inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
	inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
	inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
	inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

	# 4e
	inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
	inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
	inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
	inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
	inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
	inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
	inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
	inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3, mode='concat')
	pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

	# 5a
	inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
	inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
	inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
	inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
	inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
	inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
	inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu', name='inception_5a_pool_1_1')
	inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3, mode='concat')

	# 5b
	inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
	inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
	inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3, activation='relu', name='inception_5b_3_3')
	inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
	inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
	inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
	inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
	inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')
	print (inception_5b_output)
	pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
	print (pool5_7_7)
	pool5_7_7 = dropout(pool5_7_7, 0.4)
	print (pool5_7_7)
	layer_3b_up		= tflearn_deconv_2d(net=pool5_7_7, nb_filter=512, kernel=2, stride=2)
	layer_3b_up		= tflearn_deconv_2d(net=layer_3b_up, nb_filter=256, kernel=2, stride=2)
	layer_3b_up		= tflearn_deconv_2d(net=layer_3b_up, nb_filter=128, kernel=2, stride=2)
	layer_3b_up		= tflearn_deconv_2d(net=layer_3b_up, nb_filter=64, kernel=2, stride=2)
	layer_3b_up		= tflearn_deconv_2d(net=layer_3b_up, nb_filter=32, kernel=2, stride=2)

	# fc
	#loss = fully_connected(pool5_7_7, 17, activation='softmax')
	#loss = fully_connected(pool5_7_7, 17, activation='softmax')

	loss	= conv_2d(layer_3b_up, 2, 1, 1, activation="softmax")
	print (loss)	
	network = regression(loss, optimizer='momentum',
	                     loss='categorical_crossentropy',
	                     learning_rate=0.001)

	# to train
	model = tflearn.DNN(network, tensorboard_dir=log_dir)#checkpoint_path='model_googlenet', max_checkpoints=1, tensorboard_verbose=2)


	return model

	# model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
	#           show_metric=True, batch_size=64, snapshot_step=200,
	#           snapshot_epoch=False, run_id='googlenet_oxflowers17')


	# # Optimizer
	# regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam', loss=jacquard_2d, learning_rate=lr) # categorical_crossentropy/dice_loss_3d

	# model   = tflearn.models.dnn.DNN(regress, tensorboard_dir=log_dir)

	# # Saving the model
	# create_folder("Weights/baseline/")
	# model.save("Weights/baseline/Teko_5x5")

	# return model