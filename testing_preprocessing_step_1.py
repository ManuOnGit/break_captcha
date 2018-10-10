import numpy as np
from tools import *

img = get_random_img()
img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_par = ('kernel', [np.ones((i, i), np.uint8) for i in range(5)])
iterations_par = ('iterations', range(10))



########
# blur #
########
title = 'gaussian_blur'
transf = cv2.GaussianBlur
par_1 = ('ksize', [(i, i) for i in [1, 3, 5, 7, 9]])
par_2 = ('sigmaX', [0, 0.1, 1, 2, 3, 4, 5])
pars_fixed = {}
plotParamVar(img, img_2, title, transf, par_1, par_2, pars_fixed, PREPRO + title + '.png')



#########
# erode #
#########
title = 'erode'
transf = cv2.erode
par_1 = kernel_par
par_2 = iterations_par
pars_fixed = {}
plotParamVar(img, img_2, title, transf, par_1, par_2, pars_fixed, PREPRO + title + '.png')



##########
# dilate #
##########
title = 'dilate'
transf = cv2.dilate
par_1 = kernel_par
par_2 = iterations_par
pars_fixed = {}
plotParamVar(img, img_2, title, transf, par_1, par_2, pars_fixed, PREPRO + title + '.png')



################
# morphologyEx #
################
morph_dico = {
                'morph_open' : cv2.MORPH_OPEN,
                'morph_close' : cv2.MORPH_CLOSE,
                'morph_gradient' : cv2.MORPH_GRADIENT,
                'morph_tophat' : cv2.MORPH_TOPHAT,
                'morph_blackhat' : cv2.MORPH_BLACKHAT,
                'morph_hitmiss' : cv2.MORPH_HITMISS
            }
transf = cv2.morphologyEx
par_1 = kernel_par
par_2 = iterations_par

for title, operation in morph_dico.items():    
    pars_fixed = {'op' : operation}
    plotParamVar(img, img_2, title, transf, par_1, par_2, pars_fixed, PREPRO + title + '.png')



####################
# favorites transf #
####################

def transfP(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return transf(src = src, **pars_fixed)

##########
# dilate #
##########
title = 'many_dilate'
transf = cv2.dilate
pars_fixed = {'kernel' : np.ones((0, 0), np.uint8), 'iterations' : 1}
plotManyExamples(title, transfP, PREPRO + title + '.png')


################
# morph close #
################
title = 'many_morph_close'
transf = cv2.morphologyEx
pars_fixed = {'op' : cv2.MORPH_CLOSE, 'kernel' : np.ones((4, 4), np.uint8), 'iterations' : 1}
plotManyExamples(title, transfP, PREPRO + title + '.png')


##################
# morph gradient #
##################
title = 'many_morph_gradient'
transf = cv2.morphologyEx
pars_fixed = {'op' : cv2.MORPH_GRADIENT, 'kernel' : np.ones((4, 4), np.uint8), 'iterations' : 1}
plotManyExamples(title, transfP, PREPRO + title + '.png')


