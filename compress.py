#Code for generating compressed images.
from PIL import Image
import os	

#The compression function takes a 'quality' variable to determine compression level. 
#quality 100 is best, 95 is original, and 1 is lowest. The values below represent quality options.
compression_values = [1,3,5,7,9]
current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/compression'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

#For saving the compressed images, create a folder structure if it does not already exist.
for compression_value in compression_values:
	parent_dir = req_dir
	compressionpath = 'compression_' + str(compression_value)
	path = os.path.join(parent_dir, compressionpath)
	if not os.path.exists(path):
		os.mkdir(path)
	parent_dir = path
	for face_number in range(1,31):
		directory = str(face_number)
		path = os.path.join(parent_dir, directory)
		if not os.path.exists(path):
			os.mkdir(path)

def compress_with_given_quality(image_path, output_path, quality=10):
    img = Image.open(image_path)
    img.save(output_path, quality=quality)
    return Image.open(output_path)

if __name__ == '__main__':
	for compression_value in compression_values:
		for face_number in range(1,31):
			for face_angle_number in range(1,10):
				print('Compressing images: compression value '+str(compression_value)+', face '+str(face_number)+', face angle ' + str(face_angle_number)+'	', end='\r')
				compress_with_given_quality('images/original/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', 'images/compression/compression_'+str(compression_value)+'/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', compression_value)

print('Image compression done, shutting down script.'+' '*30)