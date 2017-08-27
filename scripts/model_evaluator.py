# Import modules 
from dataset import Dataset
from classification_model import Classification_model
from classification_model import Hyper_params,Lr,Lr_reduce_on_plateau,Early_stop,Learning_params

import keras
from keras.models import Model
from keras import backend as K

import tensorflow as tf

from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from numpy import *

import os
from shutil import rmtree
from shutil import copy2

from time import gmtime, strftime
cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# This code supports tensorflow backend only
assert K.image_dim_ordering() == 'tf'



class Model_evaluator:
    def __init__(self,clf_model, class_names, batch_size=32,
                 verbose=1, is_save_predictions=False):
        self.clf_model = clf_model
        self.class_names = class_names
        self.batch_size = batch_size
        self.verbose = verbose
        self.is_save_predictions = is_save_predictions
    
    def get_classifier_score(self, dataset):
    	score = self.clf_model.model.evaluate(dataset.images, dataset.labels,
                                         batch_size=self.batch_size, verbose=self.verbose)
        print dataset.dataset_name + ' set loss:', score[0]
        print dataset.dataset_name + ' set accuracy:', score[1]
        return score

    def evaluate(self, dataset): 
        y_pred = self.clf_model.model.predict(dataset.images)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        if self.is_save_predictions:
            np.savetxt(os.path.join(self.clf_model.summary_dir, 
                                    dataset.dataset_name + '_y_pred.csv'), y_pred, delimiter=",")
            np.savetxt(os.path.join(self.clf_model.summary_dir, 
                                    dataset.dataset_name + '_y_pred_classes.csv'), y_pred_classes, delimiter=",")
                       
        clf_report = classification_report(np.argmax(dataset.labels,axis=1),
                                           y_pred_classes,target_names=self.class_names)
        print(clf_report)
        conf = confusion_matrix(np.argmax(dataset.labels,axis=1), y_pred_classes)
        print(conf)
                       
    def save_classification_errors(self, dataset, save_dir, softmax_threshold=0.5):
        error_dir = os.path.join(save_dir, 'errors')
        print 'saving classification errors to ' + error_dir
        if os.path.exists(error_dir):
            rmtree(error_dir)
        os.makedirs(error_dir)
        for i in range(0,dataset.nb_classes):
            cur_error_class_dir = os.path.join(error_dir, self.class_names[i])
            os.makedirs(cur_error_class_dir )
            os.makedirs(cur_error_class_dir + '/false_positive')
            os.makedirs(cur_error_class_dir + '/false_negative')
            print "created " + cur_error_class_dir + '/false_positive' 
            print "created " + cur_error_class_dir + '/false_negative' 
    
        y_pred = self.clf_model.model.predict(dataset.images)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(dataset.labels,axis=1)              
        for i in range(0,len(y_pred_classes)):
            if y_pred_classes[i] != y_true_classes[i]:
                # print 'max prob = ' + str(max(y_pred[i,:]))
                if max(y_pred[i,:]) < softmax_threshold :
                    continue
                fn_save_dir = error_dir + '/' + dataset.class_names[y_true_classes[i]] + '/false_negative'
                fp_save_dir = error_dir + '/' + dataset.class_names[y_pred_classes[i]] + '/false_positive'
                copy2(dataset.img_dir + '/' + dataset.filenames[i], fn_save_dir)
                copy2(dataset.img_dir + '/' + dataset.filenames[i], fp_save_dir)
