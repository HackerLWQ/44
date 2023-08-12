import cv2

import matplotlib.pyplot as plt
import numpy as np
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0) #ms级
    cv2.destroyAllWindows
img=cv2.imread('cat.jpg')
cat=img[0:1000,0:500]
cv_show('cat',cat)