#Code that uses the deepface library to calculate similarity scores between face pictures for all possible deepace models, and saves those scores in tables for later visual processing.
#The deepface libary does NOT include the VGG-Face2 "model" (see explanation in paper), for creating the comparison tables of that "model" please run 'create_VGGFace2_comparison_tables.py'
from deepface import DeepFace
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from numpy import asarray
from numpy import save
import os

reference_angle_number = 5
test_angle_number = 5

models = ['DeepFace'] #can be extended with: ['VGG-Face', 'Dlib', 'Facenet', 'ArcFace', 'Facenet512', 'OpenFace', 'DeepID', 'DeepFace']
comparisontypes= ['brightness', 'compression', 'noise', 'resolution']

#For all the models, create a directory structure to save the comparison tables in, if it does not already exists.
for directory in models:
	current_dir = os.getcwd()
	req_dir = current_dir+'/saved_comparison_data/'+directory+'/'+str(reference_angle_number)+'-'+str(test_angle_number)
	if not os.path.exists(req_dir):
		os.mkdir(req_dir)
	for directory2 in ['brightness', 'compression', 'noise', 'resolution']:
		comparisondir = req_dir+'/'+directory2
		if not os.path.exists(comparisondir):
			os.mkdir(comparisondir)

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

super_starttime = time.perf_counter()
for modelname in models:
	print('Doing model '+modelname+' now!')
	abs_abs_starttime = time.perf_counter()

	print('Building '+modelname+' model...')
	model = DeepFace.build_model(modelname)
	print('Model built!')

	#Smol test before starting main task
	print('Running a short test of 10 comparisons...')
	starttime = time.perf_counter()

	for face_number in range(1,10):
		print(DeepFace.verify(img1_path = 'faces/original/'+str(face_number)+'/'+str(reference_angle_number)+'.jpg', img2_path = 'faces/original/1/'+str(test_angle_number)+'.jpg', detector_backend='skip', model=model)['distance'], end = '\r')

	totaltime = time.perf_counter() - starttime
	print('Total test execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

	print('STARTING THE 4 COMPARISONS, EXPECTED RUN TIME ~10 HOURS')
	for comparisontype in comparisontypes:
		abs_starttime = time.perf_counter()

		print('Using the '+modelname+' model to compare '+comparisontype+'. Check if this is correct.')

		options = []
		if (comparisontype == 'resolution'):
			options = [1024, 512, 256, 128, 64]
		elif (comparisontype == 'compression'):
			options = [9,7,5,3,1]
		elif (comparisontype == 'brightness'):
			options = [0.1,0.5,1.5,3,5]
		elif (comparisontype == 'noise'):
			options = [0.1, 0.3, 0.5, 0.7, 1];

		#do main task, start by comparing original reference angle to original test angle
		starttime = time.perf_counter()
		results = []

		print('Running originals comparison...')
		resultsarray = np.zeros((30,30))
		for i in range(1,31):
			for j in range(1,31):
				resultsarray[i-1,j-1] = DeepFace.verify(img1_path = 'faces/original/'+str(i)+'/'+str(reference_angle_number)+'.jpg', img2_path = 'faces/original/'+str(j)+'/'+str(test_angle_number)+'.jpg', detector_backend='skip', model=model)['distance']
		results.append(resultsarray)

		print('Originals comparison done!')
		totaltime = time.perf_counter() - starttime
		print('Execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

		#now compare original quality reference angle to all options for the chosen comparison type test angle
		print('Running original vs  '+comparisontype+' comparison...')
		starttime2 = time.perf_counter()
		for option in options:
			resultsarray = np.zeros((30,30))
			for i in range(1,31):
				for j in range(1,31):
					resultsarray[i-1,j-1] = DeepFace.verify(img1_path = 'faces/original/'+str(i)+'/'+str(reference_angle_number)+'.jpg', img2_path = 'faces/'+comparisontype+'/'+comparisontype+'_'+str(option)+'/'+str(j)+'/'+str(test_angle_number)+'.jpg', detector_backend='skip', model=model,)['distance']
			results.append(resultsarray)
			print(comparisontype+' level '+str(option)+' done.')
			totaltime = time.perf_counter() - starttime2
			print('Execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
			starttime2 = time.perf_counter()

		totaltime = time.perf_counter() - starttime
		print('Original vs '+comparisontype+' comparisons done! Total execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

		#save results to datafile for further processing
		data = asarray(results)
		save('saved_comparison_data/'+modelname+'/'+str(reference_angle_number)+'-'+str(test_angle_number)+'/'+comparisontype+'/original_vs_'+comparisontype+'s.npy', data)
		print('Data saved!')

		#now that the reference angle of original image quality comparisons are done, do the comparisons for the other options
		for option in options:
			print('Running the entire thing again but with reference image '+comparisontype+' level '+str(option))
			#start by comparing all options for the chosen comparison type *reference* angle to the original quality *test* angle
			starttime = time.perf_counter()
			results = []

			print('Running '+comparisontype+'s with original comparison...')
			resultsarray = np.zeros((30,30))
			for i in range(1,31):
				for j in range(1,31):
					resultsarray[i-1,j-1] = DeepFace.verify(img1_path = 'faces/'+comparisontype+'/'+comparisontype+'_'+str(option)+'/'+str(i)+'/'+str(reference_angle_number)+'.jpg', img2_path = 'faces/original/'+str(j)+'/'+str(test_angle_number)+'.jpg', detector_backend='skip', model=model)['distance']
			results.append(resultsarray)

			print(comparisontype+'s with original comparison done!')
			totaltime = time.perf_counter() - starttime
			print('Execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

			#now compare all options for the chosen comparison type *reference* angle to all options for the chosen comparison type *test* angle
			print('Running '+comparisontype+' level '+str(option)+' vs '+comparisontype+'s comparison...')
			starttime2 = time.perf_counter()
			for option in options:
				resultsarray = np.zeros((30,30))
				for i in range(1,31):
					for j in range(1,31):
						resultsarray[i-1,j-1] = DeepFace.verify(img1_path = 'faces/'+comparisontype+'/'+comparisontype+'_'+str(option)+'/'+str(i)+'/'+str(reference_angle_number)+'.jpg', img2_path = 'faces/'+comparisontype+'/'+comparisontype+'_'+str(option)+'/'+str(j)+'/'+str(test_angle_number)+'.jpg', detector_backend='skip', model=model)['distance']
				results.append(resultsarray)
				print(comparisontype+' level '+str(option)+' done.')
				totaltime = time.perf_counter() - starttime2
				print('Execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
				starttime2 = time.perf_counter()

			totaltime = time.perf_counter() - starttime
			print('Reference image '+comparisontype+' level '+str(option)+' comparisons done! Total execution time for this reference level: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
			totaltime = time.perf_counter() - abs_starttime
			print('Time elapsed so far for this model: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

			#save results to datafile for further processing
			data = asarray(results)
			save('saved_comparison_data/'+modelname+'/'+str(reference_angle_number)+'-'+str(test_angle_number)+'/'+comparisontype+'/'+comparisontype+'_'+str(option)+'_vs_'+comparisontype+'s.npy', data)
			print('Data saved!')

		totaltime = time.perf_counter() - abs_starttime
		print(comparisontype+' comparison type fully done! Total execution time for this comparison type: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

	totaltime = time.perf_counter() - abs_abs_starttime
	print('All comparison types done! This was model: '+modelname+'. Excecution time for this model: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
	totaltime = time.perf_counter() - super_starttime
	print('Time elapsed so far: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

print('Finally done! Total time elapsed: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
print('Shutting down script.')
