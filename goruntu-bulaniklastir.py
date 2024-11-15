import cv2 
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


img = cv2.imread("new-york-city.jpg")
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)


plt.figure(),plt.imshow(img),plt.axis("off"),plt.title("Original"),plt.show()   

"""""
ortalama bulaniklastirma yontemi
"""""

dst2 = cv2.blur(img, ksize = (3,3))
plt.figure(),plt.imshow(dst2),plt.axis("off"),plt.title("Ortalama Blur")

"""
Gaussian Blur
"""

gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure(),plt.imshow(gb),plt.axis("off"),plt.title("Gaussian Blur")

"""
Medyan Blur
"""
mb = cv2.medianBlur(img, ksize = 3 )
plt.figure(), plt.imshow(mb), plt.axis("off"),plt.title("Medyan Blur")

def gaussianNoise(image) :  
    row,col,ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    
    return noisy

img = cv2.imread("new-york-city.jpg")
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)/255
plt.figure(),plt.imshow(img),plt.axis("off"),plt.title("Original"),plt.show() 

gaussianNoisyImage = gaussianNoise(img)
plt.figure(),plt.imshow(gaussianNoisyImage),plt.axis("off"),plt.title("GaussNoisy"),plt.show() 

#gauss blur
gb2 = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure(),plt.imshow(gb2), plt.axis("off"), plt.title(" With Gauss Blur")

def SaltPepperNoise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    
    amount = 0.004
    
    noisy =  np.copy(image)
    
    #salt beyaz
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 1   
    
    # pepper siyah
    num_pepper = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy 


spImage = SaltPepperNoise(img)
plt.figure(),plt.imshow(spImage), plt.axis("off"), plt.title("SP Image")
    
mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize = 3 )
plt.figure(), plt.imshow(mb2), plt.axis("off"),plt.title("with Medyan Blur")