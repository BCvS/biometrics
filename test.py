import os
from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = DeepFace.detectFace(img_path='images/brightness/brightness_0.1/1/1.jpg', detector_backend = 'mtcnn', enforce_detection = False)
plt.imshow(img)
plt.show()
#C:/Users/spijk/Documents/_assignment/biometrics_data/