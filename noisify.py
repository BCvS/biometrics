import numpy as np
import os
from PIL import Image
from skimage.util import random_noise

current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/noise'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

for i in [0.1, 0.3, 0.5, 0.7, 1]:
	parent_dir = req_dir
	compressionpath = 'noise_' + str(i)
	path = os.path.join(parent_dir, compressionpath)
	os.mkdir(path)
	parent_dir = path
	for j in range(1,31):
		directory = str(j)
		path = os.path.join(parent_dir, directory)
		if not os.path.exists(path):
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
		for i in range(1,31):
			for j in range(1,10):
				noisify('images/original/'+str(i)+'/'+str(j)+'.jpg', 'images/noise/noise_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)

print("Noisify done, shutting down script.")