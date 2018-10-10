import cv2
import numpy as np

def preprocessing_step_1(img):
    img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pars_fixed = {'op' : cv2.MORPH_CLOSE, 'kernel' : np.ones((4, 4), np.uint8), 'iterations' : 1}
    return cv2.morphologyEx(src = img_2, **pars_fixed)

def preprocessing_step_2(img):
    pars_fixed = {'maxval' : 200, 'thresh' : 250, 'type' : cv2.THRESH_TOZERO}
    return cv2.threshold(src = img, **pars_fixed)[1]

def preprocessing_step_3(img):
    return cv2.resize(img, (84, 84))

def preprocessing_full(img):
    return preprocessing_step_3(preprocessing_step_2(preprocessing_step_1(img)))
