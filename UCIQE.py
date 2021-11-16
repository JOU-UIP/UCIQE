"""
UCIQE
======================================
Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576.
UCIQE= c1*var_chr+c2*con_lum+c3*aver_sat
var_chr   is σc : the standard deviation of chroma
con_lum is conl: the contrast of luminance
aver_sat  is μs : the average of saturation
coe_metric=[c1, c2, c3]are weighted coefficients.
---------------------------------------------------------
When you want to use the uciqe function, you must give the values of two parameters,
one is the nargin value you calculated and the location and name format of the image
you want to calculate the uciqe value.The format of the function is UCIQE.uciqe(nargin,loc)
---------------------------------------------------------
The input image must be RGB image
======================================
"""

import cv2
import numpy as np


def uciqe(nargin,loc):
    img_bgr = cv2.imread(loc)        # Used to read image files
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    if nargin == 1:                                 # According to training result mentioned in the paper:
        coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)
    return quality_val

