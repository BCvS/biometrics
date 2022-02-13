#This script extracts faces from images using the mtcnn algorithm and stores them as new images. 
#Doing this speeds up the comparison scripts, as they do not have to run the mtcnn algorithm multiple times for every face (every time it is compared to another face).
from deepface import DeepFace
import cv2
import time
import os

folders = ['resolution', 'compression','brightness', 'noise']

print('Creating folder structure...')
#create folder structure for faces/original/*
current_dir = os.getcwd()

req_dir = current_dir+'/faces'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = current_dir+'/original'
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
		options = [0.1, 0.3, 0.5, 0.7, 1]

	parent_dir = current_dir+'/faces/'+folder

	if not os.path.exists(parent_dir):
		os.mkdir(parent_dir)

	for option in options:
		parent_dir = current_dir+'/faces/'+folder
		subpath = folder+'_'+str(option)
		path = parent_dir+'/'+subpath
		if not os.path.exists(path):
			os.mkdir(path)
		parent_dir = path
		for j in range(1,31):
			directory = str(j)
			path = os.path.join(parent_dir, directory)
			if not os.path.exists(path):
				os.mkdir(path)
print('Done creating folder structure!')

abs_starttime = time.perf_counter()
starttime = time.perf_counter()

img_path = 'images/original/'
img_output_path = 'faces/original'

#Extract the faces using the mtcnn algorithm. If the algorithm cannot find a face, it simply outputs the same image.
print('Detecting faces in original images...')
for face_number in range(1,31):
	for face_angle_number in range(1,10):
		print('Detecting face '+str(face_number)+', angle '+str(face_angle_number)+'	', end='\r')
		img = DeepFace.detectFace(img_path=img_path+str(face_number)+'/'+str(face_angle_number)+'.jpg', detector_backend='mtcnn', enforce_detection=False)
		img = img * 255
		cv2.imwrite((img_output_path+'/'+str(face_number)+'/'+str(face_angle_number)+'.jpg'),img[:, :, ::-1])

totaltime = time.perf_counter() - starttime
print('Done with original images! Execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
starttime = time.perf_counter()

print('Detecting faces in edited images...')
for folder in folders:
	print('Doing '+folder+' values:')
	options = []
	if (folder == 'resolution'):
		options = [1024, 512, 256, 128, 64]
	elif (folder == 'compression'):
		options = [9,7,5,3,1]
	elif (folder == 'brightness'):
		options = [0.1,0.5,1.5,3,5]
	elif (folder == 'noise'):
		options = [0.1, 0.3, 0.5, 0.7, 1];

	for option in options:
		subpath = folder+'/'+folder+'_'+str(option)+'/'
		img_path = 'images/'+subpath
		img_output_path = 'faces/'+subpath

		for face_number in range(1,31):
			for face_angle_number in range(1,10):
				print('For '+folder+' level '+str(option)+', detecting face '+str(face_number)+', angle '+str(face_angle_number)+'	'*2, end='\r')
				img = DeepFace.detectFace(img_path=img_path+'/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', detector_backend='mtcnn', enforce_detection=False)
				img = img * 255
				cv2.imwrite((img_output_path+'/'+str(face_number)+'/'+str(face_angle_number)+'.jpg'),img[:, :, ::-1])

	totaltime = time.perf_counter() - starttime
	print('Done with '+folder+' pictures! Execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime)))+'	'*2)
	starttime = time.perf_counter()

totaltime = time.perf_counter() - abs_starttime
print('Completely done detecting faces! Total execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
print('Shutting down script.')
