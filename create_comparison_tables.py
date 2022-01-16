from deepface import DeepFace
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from numpy import asarray
from numpy import save

modelname = "DeepFace" #choices: [Deepface, VGG-Face, FaceNet, OpenFace]
comparisontype = "resolution" #choices: [resolution, compression, brightness, noise]

abs_starttime = time.perf_counter()

print("Using the " + modelname + " model to compare " + comparisontype + ". Check if this is correct.")

options = []
if (comparisontype == 'resolution'):
	options = [1024, 512, 256, 128, 64]
elif (comparisontype == 'compression'):
	options = [9,7,5,3,1]
elif (comparisontype == 'brightness'):
	options = [0.1,0.5,1.5,3,5]
elif (comparisontype == 'noise'):
	options == [0.1, 0.3, 0.5, 0.7, 1];

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

print("Building " + modelname + " model...")
model = DeepFace.build_model(modelname)
print("Model built!")

#Smol test before starting main task
print("Running a short test of 10 comparisons...")
starttime = time.perf_counter()

for i in range(1,10):
	print(DeepFace.verify(img1_path = "original/"+str(i)+"/5.jpg", img2_path = "original/1/1.jpg", model=model)['distance'], end = "\r")

totaltime = time.perf_counter() - starttime
print("Total test execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

#do main task, start by comparing original front to original side
starttime = time.perf_counter()
results = []

print("Running originals comparison...")
resultsarray = np.zeros((30,30))
for i in range(1,31):
	for j in range(1,31):
		#print(str(i) + " " + str(j), end = "\r")
		resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "original/"+str(i)+"/5.jpg", img2_path = "original/"+str(j)+"/1.jpg", model=model)['distance']
results.append(resultsarray)

print("Originals comparison done!")
totaltime = time.perf_counter() - starttime
print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

#Now compare original front to compressed side
print("Running original vs  "+comparisontype + " comparison...")
starttime2 = time.perf_counter()
for h in options:
	resultsarray = np.zeros((30,30))
	for i in range(1,31):
		for j in range(1,31):
			#print(str(h) + " " + str(i) + " " + str(j), end = "\r")
			resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "original/"+str(i)+"/5.jpg", img2_path = comparisontype+"/"+comparisontype+"_"+str(h)+"/"+str(j)+"/1.jpg", model=model)['distance']
	results.append(resultsarray)
	print(comparisontype + " level " + str(h) + " done.")
	totaltime = time.perf_counter() - starttime2
	print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
	starttime2 = time.perf_counter()

totaltime = time.perf_counter() - starttime
print("Original vs "+comparisontype+ " comparisons done! Total execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

#Save results to datafile for further processing
data = asarray(results)
save('saved_comparison_data/'+comparisontype+'/'+modelname+'/original_vs_'+comparisontype+'s.npy', data)
print("Data saved!")


for y in options:
	print("Running the entire thing again but with reference image "+ comparisontype+ " level "+ str(y))
	#do main task, start by comparing original front to original side
	starttime = time.perf_counter()
	results = []

	print("Running " +comparisontype+ "s with original comparison...")
	resultsarray = np.zeros((30,30))
	for i in range(1,31):
		for j in range(1,31):
			#print(str(i) + " " + str(j), end = "\r")
			resultsarray[i-1,j-1] = DeepFace.verify(img1_path = comparisontype+"/"+comparisontype+"_"+str(y)+"/"+str(i)+"/5.jpg", img2_path = "original/"+str(j)+"/1.jpg", model=model)['distance']
	results.append(resultsarray)

	print(comparisontype+"s with original comparison done!")
	totaltime = time.perf_counter() - starttime
	print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))



	#Now compare original front to compressed side
	print("Running "+comparisontype+ " level " + str(y) + " vs "+comparisontype+ "s comparison...")
	starttime2 = time.perf_counter()
	for h in options:
		resultsarray = np.zeros((30,30))
		for i in range(1,31):
			for j in range(1,31):
				#print(str(h) + " " + str(i) + " " + str(j), end = "\r")
				resultsarray[i-1,j-1] = DeepFace.verify(img1_path = comparisontype+"/"+comparisontype+"_"+str(y)+"/"+str(i)+"/5.jpg", img2_path = comparisontype+"/"+comparisontype+"_"+str(h)+"/"+str(j)+"/1.jpg", model=model)['distance']
		results.append(resultsarray)
		print(comparisontype + " level " + str(h) + " done.")
		totaltime = time.perf_counter() - starttime2
		print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
		starttime2 = time.perf_counter()

	totaltime = time.perf_counter() - starttime
	print(comparisontype +"s comparisons done! Total execution time for this round: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
	totaltime = time.perf_counter() - abs_starttime
	print("Time elapsed so far: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

	#Save results to datafile for further processing
	data = asarray(results)
	save('saved_comparison_data/'+comparisontype+'/'+modelname+'/'+comparisontype+'_'+str(y)+'_vs_'+comparisontype+'s.npy', data)
	print("Data saved!")

totaltime = time.perf_counter() - abs_starttime
print("All done! Total execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
print("Shutting down script.")