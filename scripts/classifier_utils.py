import os
import sys
import random
from shutil import move
from shutil import copy2
from shutil import rmtree
from shutil import copyfile
import numpy as np
from numpy import *
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from time import gmtime, strftime

cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    
def copy_images_to_class_dirs(source_file_path, save_dir_path, class_names):
    # create data folder and folders for each class
    if os.path.exists(save_dir_path):
        rmtree(save_dir_path)
    os.makedirs(save_dir_path)

    class_dirs = []
    for class_name in class_names:
        cur_dir = os.path.join(save_dir_path, class_name)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        class_dirs.append(cur_dir)
    
    src_file = open(source_file_path, 'rt')
    for line in src_file.readlines()[1:]:
        input_line = line.split()
        src_path = input_line[1]
        img_name = src_path.rsplit('/', 1)[-1]
        dst_path = os.path.join(save_dir_path, str(input_line[2]), img_name)                         
        copyfile(src_path, dst_path)

  
def create_experiment_directories():
    experiments_dir = 'experiments'
    model_dir = os.path.join(experiments_dir, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    weights_dir = os.path.join(model_dir,'weights')
    summary_dir = os.path.join(model_dir,'summary')

    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)

    return experiments_dir, model_dir, \
    weights_dir, summary_dir


sys.setrecursionlimit(15000)

# pre process utils
def make_square(im, new_size=224, fill_color=(0, 0, 0, 0)):
    # I think this will be slow
    x, y = im.size
    size = max(new_size, x, y)
    new_im = Image.new('RGB', (size, size), (0, 0, 0, 0))
    new_im.paste(im, ((size - x) / 2, (size - y) / 2))
    new_im = new_im.resize((new_size, new_size), Image.ANTIALIAS)
    return new_im

def pre_process_one_dir(listing, path):
    print 'pre-processing ' + path + ' dir'

    for file in listing:
        im = Image.open(path + '/' + file)   
        img = make_square(im)
        img.save(path + '/' + file, 'JPEG')
          
def preprocess_full_set(src_dir,class_names):           
    
    # load data as classes, duplicate grayscale to 3 channels
    for i in range(0,len(class_names)):
        src_class_path = src_dir + '/' + class_names[i]
        print src_class_path
        img_list = os.listdir(src_class_path)
        pre_process_one_dir(img_list,src_class_path)


val_percent = 0.15
def resize_and_train_val_split(in_dir, class_names, is_sample=False):
    BASE_DIR = in_dir
   
    print class_names
    print 'resizing images'
    preprocess_full_set(BASE_DIR,class_names)
    
    train_dir = BASE_DIR + '/train'
    val_dir = BASE_DIR + '/val'
    #test_dir = BASE_DIR + '/test'
    SETS =[train_dir,val_dir]#,test_dir]
    BACKUP_DATA_SET = BASE_DIR + '/full'


    # Delete all dirs
    # Create train/val/test and full dirs
    if os.path.exists(train_dir):
        rmtree(train_dir)
        os.makedirs(train_dir)
    if os.path.exists(val_dir):
        rmtree(val_dir)
        os.makedirs(val_dir)
    #if os.path.exists(test_dir):
    #    rmtree(test_dir)
    #   os.makedirs(test_dir)
    if os.path.exists(BACKUP_DATA_SET):
        rmtree(BACKUP_DATA_SET)
        os.makedirs(BACKUP_DATA_SET)

    # Create class dirs for each set
    for cur_set in SETS:
        for name in class_names:
            cur_class_dir = cur_set + '/' + name
            if not os.path.exists(cur_class_dir):
                os.makedirs(cur_class_dir)
                print cur_class_dir + " created"

    # Copy images from current folder classes to BACKUP folder
    # Shuffle lists and randomly divide to train/val/test
    for class_name in class_names:
        backup_class_dir = BACKUP_DATA_SET + '/' + class_name
        src_class_dir = BASE_DIR + '/' + class_name
        os.makedirs(backup_class_dir)
        img_files = os.listdir(src_class_dir)
        random.shuffle(img_files)
        if is_sample:
            count = 30
        for image_name in img_files:
            if is_sample:
                count = count-1
                if count<=0:
                    break
            image_path = os.path.join(src_class_dir, image_name)
            if (os.path.isfile(image_path)):
                copy2(image_path, backup_class_dir + '/' + image_name)
                coin = random.random()
                # print coin
                if coin < val_percent :
                    move(image_path, val_dir + '/' + class_name +'/' + image_name)
                #elif coin < val_percent + test_percent:
                #    move(image_path, test_dir + '/' + class_name +'/' + image_name)
                else:
                    move(image_path, train_dir + '/' + class_name +'/' + image_name)

    # Delete class dirs
    for name in class_names:
        rmtree(BASE_DIR + '/' + name)
        
def copy_images_to_test_dir(source_file_path, dir_name):
  
    src_file = open(source_file_path, 'rt')
    for line in src_file.readlines()[1:]:
        input_line = line.split()
        src_path = input_line[1]
        img_name = src_path.rsplit('/', 1)[-1]
        dst_path = os.path.join(dir_name, img_name)                         
        copyfile(src_path, dst_path)
        
def load_test_images(test_file, test_dir, pretrained_model):
    # create image folder
    save_dir_path = os.path.join(test_dir,'images')
    print 'Copying images to test dir'
    if os.path.exists(save_dir_path):
        rmtree(save_dir_path)
    os.makedirs(save_dir_path)


    copy_images_to_test_dir(test_file, save_dir_path)
    img_list = os.listdir(save_dir_path)
    pre_process_one_dir(img_list,save_dir_path)

    nb_samples = sum([len(files) for r, d, files in os.walk(test_dir)])
    datagen = ImageDataGenerator(featurewise_center=True)
    if pretrained_model == 'VGG19':
        data_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1, 1, 3))
        target_size = (224,224)
    else:
        print " pretrained model not supported, currently only VGG19"
        sys.exit(1)
    generator = datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=128,
                class_mode='categorical',
                shuffle=False)
                
    datagen.mean = data_mean
    X_test = np.zeros((nb_samples, target_size[0],target_size[1],3), dtype=np.float32)   
    ind = 0
    while True:
        batch_data,batch_labels  = generator.next()
        num_images_gen = len(batch_data)
        if ind + num_images_gen >= nb_samples:
            batch_data = batch_data[0:nb_samples - ind,:,:,:]
            num_images_gen = len(batch_data)
        X_test[ind : ind + num_images_gen, :,:,:] = batch_data
        ind = ind + num_images_gen
        if ind >= nb_samples:
            break
    return X_test

def generate_output_file(in_file, out_file, out_classes, class_names):
    test = open('test_dataset.txt', 'rt')
    results = open('results_test.txt', 'wt')

    results.write('ad_id, image_path, cat_id\n')
    for (line,ind) in zip(test.readlines()[1:], range(0,len(out_classes))):
        line_parts = line.split()
        results.write('{0}, {1}, {2}\n'.format(line_parts[0],line_parts[1],class_names[out_classes[ind]]))        
    results.close()
    test.close()

