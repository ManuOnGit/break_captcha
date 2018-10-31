import numpy as np

from tools import *
from preprocessing import preprocessing_basic, preprocessing_full

def preprocessing_step_1(img):
    img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pars_fixed = {'op' : cv2.MORPH_CLOSE, 'kernel' : np.ones((4, 4), np.uint8), 'iterations' : 1}
    return cv2.morphologyEx(src = img_2, **pars_fixed)

img = get_random_img()
img_2 = preprocessing_step_1(img)



##########
# thresh #
##########
thresh_dico = {
                'thresh_binary_inv' : cv2.THRESH_BINARY_INV,
                'thresh_binary' : cv2.THRESH_BINARY,
                'thresh_trunc' : cv2.THRESH_TRUNC,
                'thresh_tozero' : cv2.THRESH_TOZERO,
                'thresh_tozero_inv' : cv2.THRESH_TOZERO_INV,
                'thresh_otsu' : cv2.THRESH_OTSU,
                'thresh_binary_otsu' : cv2.THRESH_BINARY + cv2.THRESH_OTSU
                }

transf = cv2.threshold
par_1 = ('maxval', [200])
par_2 = ('thresh', list(range(0, 255, 15)) + [250, 254])

for title, type in thresh_dico.items():    
    pars_fixed = {'type' : type}
    plotParamVar(img, img_2, title, transf, par_1, par_2, pars_fixed, PREPRO + '/' + title + '.png', tresh = True)



####################
# favorites transf #
####################

title = 'many_tresh_tozero'
transf = cv2.threshold
pars_fixed = {'maxval' : 200, 'thresh' : 250, 'type' : cv2.THRESH_TOZERO}

def transfP(src):
    return transf(preprocessing_step_1(src), **pars_fixed)

plotManyExamples(title, transfP, PREPRO + '/' + title + '.png', tresh = True)
