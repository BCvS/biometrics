#This script is very similar to 'create_comparison_tables.py': it calculate similarity scores between face pictures, except for that it uses the VGG-Face2 'model'. 
#Since this implementation of calculating face similarity scores is much more efficient than using the deepface libary, it also is able to calculate similarity scores for all 9*9=81 face angle comparisons really fast,
#instead of doing just a single face angle comparison. This script still contains a lot of print statements that probably just are not necessary anymore, and disabling them would speed it up.
import os
import time
from matplotlib import pyplot
from PIL import Image
import numpy as np
from numpy import asarray
from numpy import save
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf
from tensorflow.python.client import device_lib

modelname = 'VGG-Face2'
comparisontypes= ['compression', 'resolution', 'brightness', 'noise']

#Check that GPU is available and build the CNN model
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

print('Building ' + modelname + ' model...')
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
print('Model built!')

#Smol test before starting main task
print('Running a short test of 10 comparisons...')
starttime = time.perf_counter()

imgs = []
for face_number in range(1,10):
	img = Image.open('faces/original/'+str(face_number)+'/5.jpg')
	face_array = asarray(img)
	imgs.append(face_array)

samples = asarray(imgs, 'float32')
samples = preprocess_input(samples, version=2)
embeddings = model.predict(samples)

for i in range(0,9):
	print(cosine(embeddings[0], embeddings[i]), end='\r')

totaltime = time.perf_counter() - starttime
print('Total test execution time: ' + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

#Create comparisons for all face angles
for reference in range(1,10):
	for test in range(1,10):
		#create all necessary directories if they don't exist
		directory = 'VGG-Face2'
		current_dir = os.getcwd()
		req_dir = current_dir+'/saved_comparison_data/'+directory+'/'+str(reference)+'-'+str(test)
		if not os.path.exists(req_dir):
			os.mkdir(req_dir)
		for directory2 in ['brightness', 'compression', 'noise', 'resolution']:
			comparisondir = req_dir + '/' + directory2
			if not os.path.exists(comparisondir):
				os.mkdir(comparisondir)

		print('Starting the 4 comparisons, this should go really fast...')
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

			#This is the main reason why this code is more efficient than 'create_comparison_tables.py': here we calculate all the embeddings of the current face picture set once.
			starttime2= time.perf_counter()
			results = []

			#start by loading the reference face images in memory
			print('Processing images...')
			#print('Reading reference ('+str(reference)+') originals...', end='\r')
			imgs = []
			for i in range(1,31):
				img = Image.open('faces/original/'+str(i)+'/5.jpg')
				face_array = asarray(img)
				imgs.append(face_array)

			for option in options:
				#print('Reading reference ('+str(reference)+') '+comparisontype+' '+str(option)+'...', end='\r')
				for i in range(1,31):
					img = Image.open('faces/'+comparisontype+'/'+comparisontype+'_'+str(option)+'/'+str(i)+'/'+str(reference)+'.jpg')
					face_array = asarray(img)
					imgs.append(face_array)

			#now calculate the embeddings of the images in memory
			print('Calculating reference ('+str(reference)+') embeddings...')
			samples = asarray(imgs, 'float32')
			samples = preprocess_input(samples, version=2)
			front_face_embeddings = model.predict(samples)
			print('Reference embeddings done!')

			#do the same thing again, but for the test face images
			print('Reading test ('+str(test)+') originals...', end='\r')
			imgs = []
			for i in range(1,31):
				img = Image.open('faces/original/'+str(i)+'/1.jpg')
				face_array = asarray(img)
				imgs.append(face_array)

			for option in options:
				print('Reading test ('+str(test)+') '+comparisontype+' '+str(option)+'...', end='\r')
				for i in range(1,31):
					img = Image.open('faces/'+comparisontype+'/'+comparisontype+'_'+str(option)+'/'+str(i)+'/'+str(test)+'.jpg')
					face_array = asarray(img)
					imgs.append(face_array)

			print('Calculating test ('+str(test)+') embeddings...')
			samples = asarray(imgs, 'float32')
			samples = preprocess_input(samples, version=2)
			left_face_embeddings = model.predict(samples)
			print('test embeddings done!')

			totaltime = time.perf_counter() - starttime2
			print('Execution time for embeddings: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
			starttime2 = time.perf_counter()

			#Calculate the distances between embeddings and store them in a results array. Start with the originals...
			print('Comparing originals...')
			resultsarray = np.zeros((30,30))
			for i in range(0,30):
				for j in range(0,30):
					resultsarray[i,j] = cosine(front_face_embeddings[i], left_face_embeddings[j])
			results.append(resultsarray)

			print('Originals comparison done!')
			totaltime = time.perf_counter() - starttime2
			print('Execution time: ' + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

			#... then compare original reference to 'comparisontype' test
			print('Running original vs  '+comparisontype + ' comparison...')
			starttime2 = time.perf_counter()
			for test_option_number, option in enumerate(options):
				resultsarray = np.zeros((30,30))
				for i in range(0,30):
					for j in range(0,30):
						resultsarray[i,j] = resultsarray[i,j] = cosine(front_face_embeddings[i], left_face_embeddings[j+(test_option_number+1)*30])
				results.append(resultsarray)
				print(comparisontype + ' level ' + str(option) + ' done.')
				totaltime = time.perf_counter() - starttime2
				print('Execution time: ' + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
				starttime2 = time.perf_counter()

			print('Original vs '+comparisontype+ ' comparisons done!')

			#save results to datafile for further processing
			data = asarray(results)
			save('saved_comparison_data/'+modelname+'/'+str(reference)+'-'+str(test)+'/'+comparisontype+'/original_vs_'+comparisontype+'s.npy', data)
			print('Data saved!')

			#just like in 'create_comparison_tables.py', now do the comparisons for all the other options
			for reference_option_number, reference_option in enumerate(options):
				print('Running the entire thing again but with reference image '+ comparisontype+ ' level '+ str(reference_option))
				#start by comparing all options for the chosen comparison type *reference* angle to the original quality *test* angle
				starttime2 = time.perf_counter()
				results = []

				print('Running '+comparisontype+'s with original comparison...')
				resultsarray = np.zeros((30,30))
				for i in range(0,30):
					for j in range(0,30):
						resultsarray[i,j] = resultsarray[i-1,j-1] = cosine(front_face_embeddings[i+(reference_option_number+1)*30], left_face_embeddings[j])
				results.append(resultsarray)

				print(comparisontype+'s with original comparison done!')
				totaltime = time.perf_counter() - starttime2
				print('Execution time: ' + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

				#now compare all options for the chosen comparison type *reference* angle to all options for the chosen comparison type *test* angle
				print('Running '+comparisontype+' level '+str(reference_option)+' vs '+comparisontype+'s comparison...')
				starttime2 = time.perf_counter()
				for test_option_number, test_option in enumerate(options):
					resultsarray = np.zeros((30,30))
					for i in range(0,30):
						for j in range(0,30):
							resultsarray[i,j] = resultsarray[i,j] = cosine(front_face_embeddings[i+(reference_option_number+1)*30], left_face_embeddings[j+(test_option_number+1)*30])
					results.append(resultsarray)
					print(comparisontype+' level '+str(test_option)+' done.')
					totaltime = time.perf_counter() - starttime2
					print('Execution time: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
					starttime2 = time.perf_counter()

				totaltime = time.perf_counter() - starttime2
				print('Reference image '+comparisontype+' level '+str(reference_option)+' comparisons done! Total execution time for this reference level: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
				totaltime = time.perf_counter() - abs_starttime
				print('Time elapsed so far: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

				#save results to datafile for further processing
				data = asarray(results)
				save('saved_comparison_data/'+modelname+'/'+str(reference)+'-'+str(test)+'/'+comparisontype+'/'+comparisontype+'_'+str(reference_option)+'_vs_'+comparisontype+'s.npy', data)
				print('Data saved!')

			totaltime = time.perf_counter() - abs_starttime
			print(comparisontype+' comparison type fully done! Total execution time for this comparison type: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

totaltime = time.perf_counter() - starttime
print('All comparison types done! Total time elapsed: '+str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
print('Shutting down script.')