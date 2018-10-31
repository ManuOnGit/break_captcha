import cv2
import numpy as np

def preprocessing_smooth(src):
    src_2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    pars_fixed = {'op' : cv2.MORPH_CLOSE, 'kernel' : np.ones((4, 4), np.uint8), 'iterations' : 1}
    return cv2.morphologyEx(src = src_2, **pars_fixed)

def preprocessing_background(src):
    pars_fixed = {'maxval' : 200, 'thresh' : 250, 'type' : cv2.THRESH_TOZERO}
    return cv2.threshold(src = src, **pars_fixed)[1]

def preprocessing_resize(src):
    return cv2.resize(src, (84, 84))

def preprocessing_grayscale(src):
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

def preprocessing_full(src):
    return preprocessing_resize(preprocessing_background(preprocessing_smooth(src)))

def preprocessing_basic(src):
    return preprocessing_resize(preprocessing_grayscale(src))
