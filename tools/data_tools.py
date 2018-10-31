import shutil
import string
import random
import os

import cv2
from sklearn.model_selection import train_test_split

MAIN = '/Users/ek/Desktop/perso/side/break_captcha'
MAIN_DATA = MAIN + '/dataset'

DATASET_ALL = MAIN_DATA + '/all'
DATASET_TRAIN = MAIN_DATA + '/train'
DATASET_TEST = MAIN_DATA + '/test'

GHOST_ALL = DATASET_ALL + '/ghost'
NGHOST_ALL = DATASET_ALL + '/nghost'

GHOST_TRAIN = DATASET_TRAIN + '/ghost'
NGHOST_TRAIN = DATASET_TRAIN + '/nghost'

GHOST_TEST = DATASET_TEST + '/ghost'
NGHOST_TEST = DATASET_TEST + '/nghost'

PREPRO = MAIN + '/preprocessing/plots'
MODELS = MAIN + '/models'

def id_generator(size=10, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def get_random_img(ghost=None):
    ghost = random.choice([True, False]) if ghost is None else ghost    
    path = GHOST_ALL if ghost else NGHOST_ALL
    s = path + '/img_%d.png' %random.randint(0, 27000)
    return cv2.imread(s)

def get_img(path):
    return cv2.imread(path)

def renaming(path_to_walk, new_str):
    k = 0
    for  subdir, dirs, files in os.walk(path_to_walk):
        for file in files:
            if file and file[-3:] == 'png':
                old_name = path_to_walk + '/' + file
                new_name = path_to_walk + '/' + '%s%d.png' %(new_str, k)
                if not os.path.isfile(new_name):
                    os.rename(old_name, new_name)
                    k += 1
                    if k%100==0:
                        print(k, old_name, new_name)
                else:
                    return 'new path already exists'

def full_clean_dataset():
    renaming(GHOST_ALL, 'ok_')
    renaming(GHOST_ALL, 'img_')
    renaming(NGHOST_ALL, 'ok_')
    renaming(NGHOST_ALL, 'img_')

def move(path):
    path = path.replace('/train/', '/all/').replace('/test/', '/all/')
    print(path)
    print(os.path.basename(os.path.dirname(os.path.dirname(path))))
    if os.path.basename(os.path.dirname(os.path.dirname(path))) == 'all':
        print('moved')
        img_actual_type = os.path.basename(os.path.dirname(path))
        new_path = GHOST_ALL if img_actual_type[0] == 'n' else NGHOST_ALL
        new_path += '/bug_' + id_generator() + '.png'
        shutil.move(path, new_path)
        print(path, new_path)

def cleaner():
    f = open('error_track_train.txt','r')
    data = f.read()
    f.close()
    error_track_train = eval(data)
    c = [(k, v) for k, v in error_track_train.items()]
    c.sort(key = lambda x:x[1], reverse = True)
    for item in c:
        print(item)
        image_plots.displayOne(item[0], transfP = preprocessing_full)
        move_boolean = input()
        if move_boolean == 'y':
            move(item[0])
        else:
            print('NO MOVE')

def split_data(split = 0.2):
    random.seed(42)
    if os.path.isdir(DATASET_TRAIN) or os.path.isdir(DATASET_TEST):
        print('train and test dataset folders need to be deleted')
    else:
        full_clean_dataset()
        os.mkdir(DATASET_TRAIN)
        os.mkdir(GHOST_TRAIN)
        os.mkdir(NGHOST_TRAIN)
        os.mkdir(DATASET_TEST)
        os.mkdir(GHOST_TEST)
        os.mkdir(NGHOST_TEST)
        for subdir, dirs, files in os.walk(GHOST_ALL):
            for file in files:
                if file and file[-3:] == 'png':
                    if random.random() < split:
                        shutil.copy(GHOST_ALL + '/' + file, GHOST_TEST + '/' + file)
                    else:
                        shutil.copy(GHOST_ALL + '/' + file, GHOST_TRAIN + '/' + file)
        for subdir, dirs, files in os.walk(NGHOST_ALL):
            for file in files:
                if file and file[-3:] == 'png':
                    if random.random() < split:
                        shutil.copy(NGHOST_ALL + '/' + file, NGHOST_TEST + '/' + file)
                    else:
                        shutil.copy(NGHOST_ALL + '/' + file, NGHOST_TRAIN + '/' + file)
