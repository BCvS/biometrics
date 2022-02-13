#Code for generating images with reduced resolutions.
from PIL import Image
import os

#Resolution values for image width. Default resolution is 2048. Image heigth is scaled proportionally.
resolutions = [1024, 512, 256, 128, 64]
current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/resolution'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

#For saving the scaled down images, create a folder structure if it does not already exist.
for resolution in resolutions:
	parent_dir = req_dir
	compressionpath = 'resolution_' + str(resolution)
	path = os.path.join(parent_dir, compressionpath)
	if not os.path.exists(path):
		os.mkdir(path)
	parent_dir = path
	for face_number in range(1,31):
		directory = str(face_number)
		path = os.path.join(parent_dir, directory)
		if not os.path.exists(path):
			os.mkdir(path)

def change_resolution(image_path, output_path, width):
	basewidth = width
	img = Image.open(image_path)
	wpercent = (basewidth / float(img.size[0]))
	hsize = int((float(img.size[1]) * float(wpercent)))
	img = img.resize((basewidth, hsize), Image.ANTIALIAS)
	img.save(output_path)

if __name__ == '__main__':
	for resolution in resolutions:
		for face_number in range(1,31):
			for face_angle_number in range(1,10):
				print('Downsizing resolution: resolution '+str(resolution)+', face '+str(face_number)+', face angle ' + str(face_angle_number)+'	', end='\r')
				change_resolution('images/original/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', 'images/resolution/resolution_'+str(resolution)+'/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', resolution)

print('Resolution changes done, shutting down script.'+' '*30)