import cv2
from matplotlib import pyplot as plt
import random
from tools import *

SIZE = 5

def plotParamVar(img_1, img_2, title, transf, par_1, par_2, pars_fixed, saving_path=None, tresh=False):
    """ plot variations of transf function given two parameters ranges 
    """
    par_1_name = par_1[0]
    par_2_name = par_2[0]
    n = len(par_1[1]) + 1
    m =  len(par_2[1])
    images = []
    titles = []
    for par_1_val in par_1[1]:
        for par_2_val in par_2[1]:
            pars_temp = pars_fixed.copy()
            pars_temp[par_1_name] = par_1_val
            pars_temp[par_2_name] = par_2_val
            if tresh:
                images.append(transf(src = img_2, **pars_temp)[1])
            else:
                images.append(transf(src = img_2, **pars_temp))
            titles.append('%s: %s, %s: %s ' %(par_1_name, str(par_1_val), par_2_name, str(par_2_val)))
    plt.subplot(n, m, 1)
    plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    plt.title('Original picture', size = SIZE, pad=1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(n, m, 2)
    plt.imshow(img_2, 'gray')
    plt.title('Processed picture', size = SIZE, pad=1)
    plt.xticks([]),plt.yticks([])
    for i in range(len(images)):
        plt.subplot(n, m, i + m + 1);
        plt.imshow(images[i], 'gray');
        plt.title(titles[i], size = SIZE, pad=1);
        plt.xticks([]);
        plt.yticks([]);
    plt.suptitle(title, size = 10)
    if saving_path:
        figure = plt.gcf()
        figure.set_size_inches(24, 18)
        plt.savefig(saving_path, dpi = 100)
    else:
        plt.show()

def plotManyExamples(title, transfP, saving_path=None, tresh=False):
    """ plot n * m / 2 examples of a given transformation
    """
    n = 10
    m = 10
    images = []
    titles = []
    for i in range(int(n * m / 2)):
        img = get_random_img()
        images.append(img)
        if tresh:
            images.append(transfP(src = img)[1])
        else:
            images.append(transfP(src = img))
    for i in range(len(images)):
        plt.subplot(n, m, i + 1);
        plt.imshow(images[i], 'gray');
        plt.xticks([]);
        plt.yticks([]);
    plt.suptitle(title, size = 10)
    if saving_path:
        figure = plt.gcf()
        figure.set_size_inches(24, 18)
        plt.savefig(saving_path, dpi = 100)
    else:
        plt.show()
