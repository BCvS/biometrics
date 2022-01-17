from deepface import DeepFace
import cv2
import time
import os

folders = ['resolution', 'compression', 'brightness', 'noise']

print("Creating folder structure...")
#create folder structure for faces/original/*
current_dir = os.getcwd()

req_dir = current_dir + '/faces/original'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

for i in range(1,31):
	directory = str(i)
	path = os.path.join(req_dir, directory)
	if not os.path.exists(path):
		os.mkdir(path)

#create folder structure for faces/*
for folder in folders:
	options = []
	if (folder == 'resolution'):
		options = [1024, 512, 256, 128, 64]
	elif (folder == 'compression'):
		options = [9,7,5,3,1]
	elif (folder == 'brightness'):
		options = [0.1,0.5,1.5,3,5]
	elif (folder == 'noise'):
		options == [0.1, 0.3, 0.5, 0.7, 1];

	parent_dir = current_dir +'/faces/'+ folder
	if not os.path.exists(path):
		os.mkdir(parent_dir)

	for option in options:
		parent_dir = current_dir +'/faces/'+ folder
		subpath = folder + '_' + str(option)
		path = parent_dir + '/' + subpath
		if not os.path.exists(path):
			os.mkdir(path)
		parent_dir = path
		for j in range(1,31):
			directory = str(j)
			path = os.path.join(parent_dir, directory)
			if not os.path.exists(path):
				os.mkdir(path)
print("Done creating folder structure!")

abs_starttime = time.perf_counter()
starttime = time.perf_counter()

img_path = 'images/original/'
img_output_path = 'faces/original'

for i in range(1,31):
	for j in range(1,10):
		img = DeepFace.detectFace(img_path=img_path+str(i)+'/'+str(j) +'.jpg', detector_backend = 'mtcnn')
		img = img * 255
		cv2.imwrite((img_output_path + '/' +str(i)+ '/' + str(j)+'.jpg'),img[:, :, ::-1])
		print('Saved image '+ str(i) + '!')

totaltime = time.perf_counter() - starttime
print("Done with original pictures! Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
starttime = time.perf_counter()

for folder in folders:
	options = []
	if (folder == 'resolution'):
		options = [1024, 512, 256, 128, 64]
	elif (folder == 'compression'):
		options = [9,7,5,3,1]
	elif (folder == 'brightness'):
		options = [0.1,0.5,1.5,3,5]
	elif (folder == 'noise'):
		options == [0.1, 0.3, 0.5, 0.7, 1];

	for option in options:
		subpath = folder + '/' + folder + '_' + str(option) + '/'
		img_path = 'images/' + subpath
		img_output_path = 'faces' + subpath

		for i in range(1,31):
			for j in range(1,10):
				img = DeepFace.detectFace(img_path=img_path+str(i)+'/'+str(j)+'.jpg', detector_backend = 'mtcnn')
				img = img * 255
				cv2.imwrite((img_output_path +str(i)+'/'+str(j)+'.jpg'),img[:, :, ::-1])
				print('Saved image!')

	totaltime = time.perf_counter() - starttime
	print("Done with " + folder + " pictures! Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
	starttime = time.perf_counter()

totaltime = time.perf_counter() - abs_starttime
print("Completely done creating faces! Total execution time: " +  str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
pritn("Shutting down script.")
