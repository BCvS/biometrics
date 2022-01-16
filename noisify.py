import numpy as np
import os
from PIL import Image
from skimage.util import random_noise

for i in [0.1, 0.3, 0.5, 0.7, 1]:
	parent_dir = "C:/Users/spijk/Documents/_assignment/biometrics_data/noise"
	compressionpath = 'noise_' + str(i)
	path = os.path.join(parent_dir, compressionpath)
	os.mkdir(path)
	parent_dir = path
	for j in range(1,31):
		directory = str(j)
		path = os.path.join(parent_dir, directory)
		os.mkdir(path)
print("Directories generated")

def noisify(image_path, output_path, strength):
	img = Image.open(image_path)
	img_arr = np.asarray(img)
	noise_img = random_noise(img_arr, mode='gaussian', var=(strength**2))
	noise_img = (255*noise_img).astype(np.uint8)
	img = Image.fromarray(noise_img)
	img.save(output_path)

if __name__ == '__main__':
	for h in [0.1, 0.3, 0.5, 0.7, 1]:
		image_path_basis = "C:/Users/spijk/Documents/_assignment/biometrics_data/noise"
		output_path_basis = os.path.join("C:/Users/spijk/Documents/_assignment/biometrics_data/noise", "noise_" + str(h))
		for i in range(1,31):
			for j in range(1,10):
				noisify('original/'+str(i)+'/'+str(j)+'.jpg', 'noise/noise_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)

print("Noisify done, shutting down script.")