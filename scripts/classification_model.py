# Import modules 
from dataset import Dataset

# Keras modules
import keras
from keras.callbacks import History 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Input, \
						 Convolution2D, MaxPooling2D, ZeroPadding2D,Activation
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Model, model_from_json
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

import tensorflow as tf

from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
from numpy import *
# import math

import os
from collections import OrderedDict
import h5py
import json

from time import gmtime, strftime



# This code supports tensorflow backend only
assert K.image_dim_ordering() == 'tf'

cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())


class Lr:
	def __init__(self, init=0.0, decay=0.0):
		self.init = init
		self.decay = decay
		
class Hyper_params:
	def __init__(self, lr=Lr(), optimization_method=str(), momentum=0.0,
				weight_decay=0.0000, dropout_rate=0.0, freeze_layer_id=0, 
				architecture_head_type=str(), is_use_batch_normalization=None, 
				nb_filters_conv_1_1_first=0, nb_filters_conv_1_1_second=0,
				fc_layer_size_1=0, fc_layer_size_2=0 ):
		#Hyperparams
		self.lr = lr
		#optimization
		self.optimization_method = optimization_method
		self.momentum = momentum
		#regularization
		self.weight_decay = weight_decay
		self.dropout_rate = dropout_rate
		# Architecture
		self.freeze_layer_id = freeze_layer_id
		self.architecture_head_type = architecture_head_type
		self.is_use_batch_normalization = is_use_batch_normalization
		self.nb_filters_conv_1_1_first = nb_filters_conv_1_1_first
		self.nb_filters_conv_1_1_second = nb_filters_conv_1_1_second
		self.fc_layer_size_1 = fc_layer_size_1
		self.fc_layer_size_2 = fc_layer_size_2

class Early_stop:
	def __init__(self, is_use=False, min_delta=0, patience=0):
		self.is_use = is_use
		self.min_delta = min_delta
		self.patience = patience

class Lr_reduce_on_plateau:
	def __init__(self, is_use=False, factor=0, min_lr=0, patience=3):
		self.is_use = is_use
		self.factor = factor
		self.min_lr = min_lr
		self.patience = patience


class Learning_params:
	def __init__(self, nb_train_samples=0, nb_validation_samples=0, batch_size=0, 
				nb_max_epoch=0, nb_samples_per_epoch=0, nb_validation_samples_per_epoch=0, 
				ckpt_period=0, early_stop=Early_stop(), lr_reduce_on_plateau=Lr_reduce_on_plateau()):
		self.nb_train_samples = nb_train_samples
		self.nb_validation_samples = nb_validation_samples
		self.batch_size = batch_size
		self.nb_max_epoch = nb_max_epoch
		self.nb_samples_per_epoch = nb_samples_per_epoch
		self.nb_validation_samples_per_epoch = nb_validation_samples_per_epoch
		self.ckpt_period = ckpt_period
		self.early_stop = early_stop
		self.lr_reduce_on_plateau = lr_reduce_on_plateau
		

class Classification_model:
	def __init__(self,nb_classes,
				 weights_dir, summary_dir,
				 pretrained_network_name,include_top):
			self.nb_classes = nb_classes
			self.weights_dir = weights_dir
			self.summary_dir = summary_dir
			self.base_model = None
			self.load_pretrained_network(pretrained_network_name,include_top) 
	
	def load_pretrained_network(self, pretrained_network_name, include_top):            
		self.pretrained_network_name = pretrained_network_name #vgg19 or resnet
		self.include_top = include_top
		if self.pretrained_network_name == 'VGG19':
			print 'loading vgg 19 network'
			self.base_model = keras.applications.vgg19.VGG19(
				include_top=self.include_top,weights='imagenet')
			
		elif self.pretrained_network_name == 'ResNet50':
			self.base_model = keras.applications.resnet50.ResNet50(
				include_top=self.include_top,weights='imagenet')
		else:
			print 'Error - Pretrained network not supported'
			exit(1)
		self.freeze_all_layers_base_model()
		self.base_model.summary()   

	
	def freeze_all_layers_base_model(self):
		for layer in self.base_model.layers:
			layer.trainable = False    
		
		
	def set_trainable_layers_and_regularization_base_model(self):
		self.freeze_all_layers_base_model()
		for layer in self.base_model.layers[self.freeze_layer_id :]:
			layer.trainable = True
			layer.W_regularizer = keras.regularizers.get(self.w_reg)
			layer.b_regularizer = keras.regularizers.get(self.w_reg)
	
	def set_optimization_method(self):
		if self.optimization_method == 'SGD':
			self.optimizer = SGD(lr=self.lr.init, momentum=self.momentum, decay=lr.decay)
		elif self.optimization_method == 'RMSprop':
			self.optimizer = RMSprop(lr=self.lr.init)
		elif self.optimization_method == 'Adam':
			self.optimizer = Adam(lr=self.lr.init)
		else:
			print 'Error - optimization method not supported'
			exit(1)
		
	
	def set_hyper_params(self,hyper_params):
		# lr
		self.lr = hyper_params.lr
		#optimization
		self.optimization_method = hyper_params.optimization_method
		self.momentum = hyper_params.momentum
		#regularization
		self.weight_decay = hyper_params.weight_decay
		self.w_reg = keras.regularizers.l2(self.weight_decay)
		self.dropout_rate = hyper_params.dropout_rate
		# Architecture
		self.freeze_layer_id = hyper_params.freeze_layer_id
		self.architecture_head_type = hyper_params.architecture_head_type
		self.is_use_batch_normalization = hyper_params.is_use_batch_normalization 
		self.nb_filters_conv_1_1_first = hyper_params.nb_filters_conv_1_1_first
		self.nb_filters_conv_1_1_second = hyper_params.nb_filters_conv_1_1_second
		self.fc_layer_size_1 = hyper_params.fc_layer_size_1
		self.fc_layer_size_2 = hyper_params.fc_layer_size_2
		
	
	def set_loss_function(self, loss = 'None'):
		# Implement special loss  
		if loss == 'None':
			self.loss = 'categorical_crossentropy'
		else:
			self.loss = loss
	
	def build_model(self, input_shape, hyper_params):
		self.set_hyper_params(hyper_params)
		self.set_trainable_layers_and_regularization_base_model()
		self.set_optimization_method()
		self.set_loss_function()
		
		input = Input(shape=(input_shape[0],input_shape[1],3),name = 'image_input') 
		out_imagenet = self.base_model(input)
		x = out_imagenet
		if self.architecture_head_type == 'none':
			#  Enables use different image sizes - instead of resizing all images                
			x = GlobalAveragePooling2D(name='global_average_pool')(x)
			x = Flatten(name='flatten')(x)

		if self.architecture_head_type == 'no_layers': 
		#     conv 1*1
			x = Convolution2D(self.nb_filters_conv_1_1_first, 1, 1, activation='relu', init='he_normal', \
							 W_regularizer=self.w_reg, b_regularizer=self.w_reg, name='bottlekneck_1_1_conv_1')(x)
			if self.is_use_batch_normalization:
				x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
			x = Convolution2D(self.nb_filters_conv_1_1_second, 1, 1, activation='relu', init='he_normal', \
							  W_regularizer=self.w_reg, b_regularizer=self.w_reg, name='bottlekneck_1_1_conv_2')(x)
			if self.is_use_batch_normalization:
				x = keras.layers.BatchNormalization(name='batch_norm_2')(x)

			x = Flatten(name='flatten')(x)
			x = Dropout(self.dropout_rate,name='dropout')(x)

		elif self.architecture_head_type == 'one_fc_layer':
			# Add last layers 
		#     conv 1*1
			x = Convolution2D(self.nb_filters_conv_1_1_first, 1, 1, activation='relu', init='he_normal', \
							 W_regularizer=self.w_reg, b_regularizer=self.w_reg, name='bottlekneck_1_1_conv_1')(x)
			if self.is_use_batch_normalization:
				x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
		#     fc layer
			x = Flatten(name='flatten')(x)
			x = Dense(self.fc_layer_size_1, activation='relu', init='he_normal', \
					  W_regularizer=self.w_reg, b_regularizer=self.w_reg, name='fc_new_1')(x)
			x = Dropout(self.dropout_rate,name='dropout')(x)

		elif self.architecture_head_type == 'two_fc_layer':
		#     conv 1*1
			x = Convolution2D(self.nb_filters_conv_1_1_first, 1, 1, activation='relu', init='he_normal', \
							 W_regularizer=self.w_reg, b_regularizer=self.w_reg, name='bottlekneck_1_1_conv_1')(x)
			if self.is_use_batch_normalization:
				x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
		#     fc layer
			x = Flatten(name='flatten')(x)
			x = Dense(self.fc_layer_size_1, activation='relu', init='he_normal', \
					  W_regularizer=self.w_reg, b_regularizer=self.w_reg, name='fc_new_1')(x)
			x = Dropout(self.dropout_rate,name='dropout')(x)
			x = Dense(self.fc_layer_size_2, activation='relu', init='he_normal', \
					  W_regularizer=self.w_reg, b_regularizer=self.w_reg, name='fc_new_2')(x)
			x = Dropout(self.dropout_rate,name='dropout2')(x)
		else:
			print 'Error - architecture head  not supported'
			exit(1)
			
		predictions = Dense(self.nb_classes, activation='softmax',W_regularizer=self.w_reg, \
							b_regularizer=self.w_reg,name='predictions')(x)
		self.model = Model(input=input, output=predictions)        
	
	def delete_model(self):
		del self.model
		del self.history
		import gc
		gc.collect()

	def set_learning_params(self, learning_params):
		self.nb_train_samples = learning_params.nb_train_samples
		self.nb_validation_samples = learning_params.nb_validation_samples
		self.batch_size = learning_params.batch_size
		self.nb_max_epoch = learning_params.nb_max_epoch
		self.nb_samples_per_epoch = self.nb_train_samples
		self.nb_validation_samples_per_epoch = self.nb_validation_samples
		self.ckpt_period = learning_params.ckpt_period
		self.early_stop = learning_params.early_stop
		self.lr_reduce_on_plateau = learning_params.lr_reduce_on_plateau
		
	def add_callbacks(self,is_save_ckpts, is_production=False):

		callbacksList = []

		self.history = History()
		callbacksList.append(self.history)

		if not is_production:
			if is_save_ckpts:
				weights_save_path = os.path.join(
					self.weights_dir, '_weights-improvement-{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}_' \
							 + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.hdf5')
				tensorboard_callback = TensorBoard(
					log_dir=os.path.join(self.summary_dir,'./logs'))    
				checkpoint_callback = ModelCheckpoint(
								 weights_save_path, monitor='val_acc', verbose=1, save_best_only=True, \
								 save_weights_only=False, mode='min',period=self.ckpt_period)
				callbacksList.append(checkpoint_callback)
				callbacksList.append(tensorboard_callback)
			if self.early_stop.is_use:
				early_stopping_callback = EarlyStopping(
						monitor='val_acc', min_delta=self.early_stop.min_delta, \
						patience=self.early_stop.patience, verbose=1, mode='min')
				callbacksList.append(early_stopping_callback)
			if self.lr_reduce_on_plateau.is_use:
				lr_reduce_on_plateau_callback = ReduceLROnPlateau(
						monitor='val_acc', factor=self.lr_reduce_on_plateau.factor,
						verbose=1, mode='min', epsilon=0.01,
						min_lr=self.lr_reduce_on_plateau.min_lr,
						patience=self.lr_reduce_on_plateau.patience)
				callbacksList.append(lr_reduce_on_plateau_callback)
		
#         missinglink_callback = KerasCallback(owner_id="5ead2bf3-6e76-c6cb-0b9d-222759f4683d", 
#                                              project_token="yhvhhgYiyvrkCxBy")
#         callbacksList.append(missinglink_callback)
		
		
		
		self.callbacks = callbacksList
		
	def compile_model(self, learning_params, is_save_ckpts_and_plot_graphs,
					  is_production=False, pretrained_weights_filename=None):
		self.is_save_ckpts_and_plot_graphs = is_save_ckpts_and_plot_graphs
		self.set_learning_params(learning_params)
		self.add_callbacks(is_save_ckpts_and_plot_graphs, is_production)
		
		if pretrained_weights_filename != None:
			self.model.load_weights(os.path.join(self.weights_dir, pretrained_weights_filename))
			print 'loaded pretrained model: ' + pretrained_weights_filename 
			
		self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
		#save model architecture to json file
		model_json = self.model.to_json()
		with open(os.path.join(self.summary_dir,"model.json"), "w") as json_file:
			json_file.write(model_json)

		if(is_save_ckpts_and_plot_graphs):
			self.summarize_model_info()




	def load_full_model(json_file,weights_file):
		with open(os.path.join(self.summary_dir,json_file), 'r') as fp:
			self.model = model_from_json(fp.read())
		model.load_weights(os.path.join(self.weights_dir, weights_file))
	

	def train(self, train_data, train_labels,
			  validation_data=None, validation_labels=None,
			  class_weights=None, is_production=False):
		if validation_data is None or validation_labels is None:
			val_input = None
		else:
			val_input = (validation_data, validation_labels)
		try:
			self.model.fit(train_data, train_labels,
					nb_epoch=self.nb_max_epoch, batch_size=self.batch_size,
					validation_data=val_input,
					callbacks=self.callbacks, verbose=(is_production or self.is_save_ckpts_and_plot_graphs),
					class_weight=class_weights)
			if self.is_save_ckpts_and_plot_graphs:
				self.plot_train_summary()
		
		except KeyboardInterrupt:
			if self.is_save_ckpts_and_plot_graphs:
				self.plot_train_summary()
	
	def plot_train_summary(self):
		print 'Training summary'
		print(self.history.history.keys())
		print(self.history.history.values())
		# summarize history for accuracy
		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
		plt.title('model acc')
		plt.ylabel('acc')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(os.path.join(self.summary_dir,'model_acc' +
								 strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.jpg'))
		plt.show()
		# summarize history for loss
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.savefig(os.path.join(self.summary_dir,'model_loss_ ' +
								 strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.jpg'))
		plt.show()
  
		
	def summarize_model_info(self):
		self.info = OrderedDict()
		self.info['nb_classes'] = self.nb_classes
		self.info['pretrained_network_name'] = self.pretrained_network_name
		self.info['loss'] = self.loss
		#hyperparams
		self.info['lr_init'] = self.lr.init
		self.info['lr_decay'] = self.lr.decay
		self.info['optimization_method'] = self.optimization_method
		self.info['momentum'] = self.momentum
		self.info['weight_decay'] = self.weight_decay
		#learning params
		self.info['nb_train_samples'] = self.nb_train_samples
		self.info['batch_size'] = self.batch_size
		self.info['nb_train_samples'] = self.nb_train_samples
		self.info['nb_max_epoch'] = self.nb_max_epoch
		self.info['nb_samples_per_epoch'] = self.nb_samples_per_epoch
		self.info['nb_validation_samples_per_epoch'] = self.nb_validation_samples_per_epoch
		self.info['ckpt_period'] = self.ckpt_period
		self.info['early_stop_is_use'] = self.early_stop.is_use
		self.info['early_stop_min_delta'] = self.early_stop.min_delta
		self.info['early_stop_patience'] = self.early_stop.patience
		self.info['lr_reduce_on_plateau_is_use'] = self.lr_reduce_on_plateau.is_use
		self.info['lr_reduce_on_plateau_factor'] = self.lr_reduce_on_plateau.factor
		self.info['lr_reduce_on_plateau_min_lr'] = self.lr_reduce_on_plateau.min_lr
		#architecture params
		self.info['freeze_layer_id'] = self.freeze_layer_id
		self.info['architecture_head_type'] = self.architecture_head_type
		self.info['is_use_batch_normalization'] = self.is_use_batch_normalization 
		self.info['nb_filters_conv_1_1_first'] = self.nb_filters_conv_1_1_first
		self.info['nb_filters_conv_1_1_second'] = self.nb_filters_conv_1_1_second
		self.info['fc_layer_size_1'] = self.fc_layer_size_1
		self.info['fc_layer_size_2'] = self.fc_layer_size_2
		self.info['layer_names'] = str()
		for i, layer in enumerate(self.model.layers):
			self.info['layer_names'] = str(i) + ': ' + layer.name
			print i, layer.name
		self.info['model_summary'] = self.model.summary()
		
		with open(os.path.join(self.summary_dir, 'model_summary.txt'),'w') as f:
			for k, v in self.info.items():
				f.write(k + ' : ' + str(v) + '\n')     
