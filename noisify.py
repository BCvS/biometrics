#Code for generating images with added noise.
import numpy as np
import os
from PIL import Image
from skimage.util import random_noise

#For generating noise, a Gaussian distribution is used. The variance of the distribution is manually set. 
#The values below represent the standard deviation options used to calculate said variance (variance = (standard deviation)**2)
noise_values = [0.1, 0.3, 0.5, 0.7, 1]
current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/noise'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

#For saving the noisified images, create a folder structure if it does not already exist.
for noise_value in noise_values:
	parent_dir = req_dir
	compressionpath = 'noise_' + str(noise_value)
	path = os.path.join(parent_dir, compressionpath)
	if not os.path.exists(path):
		os.mkdir(path)
	parent_dir = path
	for face_number in range(1,31):
		directory = str(face_number)
		path = os.path.join(parent_dir, directory)
		if not os.path.exists(path):
			os.mkdir(path)

def noisify(image_path, output_path, strength):
	img = Image.open(image_path)
	img_arr = np.asarray(img)
	noise_img = random_noise(img_arr, mode='gaussian', var=(strength**2))
	noise_img = (255*noise_img).astype(np.uint8)
	img = Image.fromarray(noise_img)
	img.save(output_path)

if __name__ == '__main__':
	for noise_value in noise_values:
		for face_number in range(1,31):
			for face_angle_number in range(1,10):
				print('Adding noise: noise variance '+str(noise_value)+', face '+str(face_number)+', face angle ' + str(face_angle_number)+'	', end='\r')
				noisify('images/original/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', 'images/noise/noise_'+str(noise_value)+'/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', noise_value)

print('Noisify done, shutting down script.'+' '*30)