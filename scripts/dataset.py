# Import modules 

import keras
from keras.callbacks import History 
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Input, Convolution2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import random
from shutil import rmtree
import os
import sys
import collections 
from time import gmtime, strftime
cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())






class Dataset:
    def __init__(self, data_dir, dataset_name, pretrained_model,
                class_names=None,
                is_resize=False,target_size=(224,224),
                augment_factor=1.0,
                fliplr=False, rotate_range=0,
                shift_range=0, zoom_range=0,
                luminance_range=0, is_production=False):
        self.is_production = is_production
        self.img_dir = data_dir
        if not self.is_production:
            self.img_dir = os.path.join(data_dir, dataset_name)
        print self.img_dir
        self.sample_dir = os.path.join(self.img_dir,'sample')
        if os.path.exists(self.sample_dir):
            rmtree(self.sample_dir)
        self.pretrained_model = pretrained_model
        if self.pretrained_model == 'VGG19':
            self.data_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1, 1, 3))
        else:
            print " pretrained model not supported, currently only VGG19"
            sys.exit(1)


        if class_names != None:
            self.class_names = class_names
        else:
            self.class_names = None
        self.target_size = target_size
        self.images = None
        self.labels = None
        self.info = dict()
        self.nb_samples = 0
        self.nb_unique_samples = 0
        self.nb_classes = 0
        self.dataset_name = dataset_name
        self.is_train_set = self.dataset_name == 'train'
        self.augment_factor = augment_factor
        self.load_and_augment_images(fliplr, rotate_range,
                 shift_range, zoom_range, luminance_range)
        if is_resize:
            self.resize_images(resize_size)
    
    def create_sample_dir(self):
        print 'Creating sample dir at ' + self.sample_dir
        os.makedirs(self.sample_dir)
        for name in self.class_names:
            cur_class_dir = self.sample_dir + '/' + name
            if not os.path.exists(cur_class_dir):
                os.makedirs(cur_class_dir)
                print cur_class_dir + " created"
    
    def save_sample_for_visualization(self):
        self.create_sample_dir()
        for cur_class in range(0,self.nb_classes):
            count = 5
            while count > 0:
                index = random.randint(0,self.nb_samples)
                if self.labels[index,cur_class] == 1:
                    scipy.misc.toimage(self.images[index,:,:,:], cmin=0.0). \
                    save(os.path.join(self.sample_dir, self.class_names[cur_class], 'sample' + str(count) + '.jpg'))
                    count = count - 1
    
    def get_class_weights(self,classes_support):
        counter = collections.Counter(classes_support)
        majority = max(counter.values())
        classes_weights = {cls: float(majority/count) for cls, count in counter.items()}
        max_weight = max(classes_weights.values())
        for key in classes_weights:
            classes_weights[key] = classes_weights[key] / max_weight
        return  classes_weights


    def load_and_augment_images(self, fliplr, rotate_range,
                 shift_range, zoom_range, luminance_range):
        self.nb_unique_samples = sum([len(files) for r, d, files in os.walk(self.img_dir)])
        self.nb_samples = int(self.nb_unique_samples * self.augment_factor)
        if self.class_names == None:
            self.class_names = [ item for item in os.listdir(self.img_dir)
                               if os.path.isdir(os.path.join(self.img_dir, item))]
        self.classes_support = dict()
        for i in range(0,len(self.class_names)):
            num_images =  len([name for name in os.listdir(os.path.join(self.img_dir,self.class_names[i]))])
            self.classes_support[i] = num_images
        self.classes_weights = self.get_class_weights(self.classes_support)

        self.nb_classes = len(self.class_names)
        print  str(self.nb_classes) + ' Classes detected:'
        print self.class_names
        print 'Loading ' + str(self.nb_unique_samples) + ' unique images from ' + self.img_dir +'/ directory'
        self.info['class_names'] = self.class_names
        
        if self.is_train_set:
#             TODO: add luminance_range augmentation
            datagen = ImageDataGenerator(featurewise_center=True, horizontal_flip=fliplr,
            width_shift_range=shift_range, height_shift_range=shift_range,
            zoom_range=zoom_range)
        else:
            datagen = ImageDataGenerator(featurewise_center=True)
        generator = datagen.flow_from_directory(
                self.img_dir,
                target_size=self.target_size,
                batch_size=32,
                class_mode='categorical',
                shuffle=self.is_train_set,
                classes=self.class_names)
        datagen.mean = self.data_mean
        images = np.zeros((self.nb_samples, self.target_size[0],self.target_size[1],3), dtype=np.float32)   
        labels = np.zeros((self.nb_samples,self.nb_classes))
        ind = 0
        while True:
            batch_data, batch_labels  = generator.next()
            num_images_gen = len(batch_data)
            if ind + num_images_gen >= self.nb_samples:
                batch_data = batch_data[0:self.nb_samples - ind,:,:,:]
                batch_labels = batch_labels[0:self.nb_samples - ind,:]
                num_images_gen = len(batch_data)
            images[ind : ind + num_images_gen, :,:,:] = batch_data
            labels[ind : ind + num_images_gen, :] = batch_labels
            ind = ind + num_images_gen
            if ind >= self.nb_samples:
                break

        self.images = images
        self.labels = labels
        if self.is_train_set:
            self.filenames = None # Not the same order if shuffle is true
        else:    
            self.filenames = generator.filenames
        # self.save_sample_for_visualization()
        print 'loaded and augmented all images'
        print 'train shape: ' + str(self.images.shape)
    
        self.info['img_size'] = self.target_size
        self.info['nb_unique_samples'] = self.nb_unique_samples
        self.info['nb_samples'] = self.nb_samples

        if self.is_train_set:
            self.info['aug_info'] = 'fliplr: ' + str(fliplr) + '\nrotate_range: ' + \
            str(rotate_range) + '\nshift_range: ' + str(shift_range) + '\nzoom_range: ' + str(zoom_range)
    
    def resize_images(resize_size):
        print 'no implementation for resize method'      
