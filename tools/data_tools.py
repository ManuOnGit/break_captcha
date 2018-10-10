import cv2
import os, sys
import random
from sklearn.model_selection import train_test_split

GHOST = '/Users/ek/Desktop/perso/side/break_captcha/dataset/ghost/'
NGHOST = '/Users/ek/Desktop/perso/side/break_captcha/dataset/nghost/'
PREPRO = '/Users/ek/Desktop/perso/side/break_captcha/preprocessing/plots/'

def get_random_img(ghost=None):
    ghost = random.choice([True, False]) if ghost is None else ghost    
    path = GHOST if ghost else NGHOST
    s = path + 'img_%d.png' %random.randint(0, 27000)
    return cv2.imread(s)

def get_img(path):
    return cv2.imread(path)

def renaming(path_to_walk, new_str):
    k = 0
    for  subdir, dirs, files in os.walk(path_to_walk):
        for file in files:
            if file and file[-3:] == 'png':
                old_name = path_to_walk + file
                new_name = path_to_walk + '%s%d.png' %(new_str, k)
                if not os.path.isfile(new_name):
                    os.rename(old_name, new_name)
                    k += 1
                    if k%100==0:
                        print(k, old_name, new_name)
                else:
                    return 'new path already exists'
