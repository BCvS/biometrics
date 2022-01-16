from deepface import DeepFace
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from numpy import asarray
from numpy import save


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

print("Building model...")
model = DeepFace.build_model("DeepFace")
print("Model built!")

#Smol test before starting main task
print("Running test of 10 comparisons...")
starttime = time.perf_counter()

for i in range(1,10):
	print(DeepFace.verify(img1_path = "original/"+str(i)+"/5.jpg", img2_path = "original/1/1.jpg", model=model)['distance'], end = "\r")

totaltime = time.perf_counter() - starttime
print("Total test execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

# #do main task, start by comparing original front to original side
# starttime = time.perf_counter()
# results = []

# print("Running originals comparison...")
# resultsarray = np.zeros((30,30))
# for i in range(1,31):
# 	for j in range(1,31):
# 		#print(str(i) + " " + str(j), end = "\r")
# 		resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "original/"+str(i)+"/5.jpg", img2_path = "original/"+str(j)+"/1.jpg", model=model)['distance']
# results.append(resultsarray)

# print("Originals comparison done!")
# totaltime = time.perf_counter() - starttime
# print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

# #Now compare original front to compressed side
# print("Running original vs compression comparison...")
# starttime2 = time.perf_counter()
# for h in [1,3,5,7,9]:
# 	resultsarray = np.zeros((30,30))
# 	for i in range(1,31):
# 		for j in range(1,31):
# 			#print(str(h) + " " + str(i) + " " + str(j), end = "\r")
# 			resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "original/"+str(i)+"/5.jpg", img2_path = "compression/compression_"+str(h)+"/"+str(j)+"/1.jpg", model=model)['distance']
# 	results.append(resultsarray)
# 	print("Compression level " + str(h) + " done.")
# 	totaltime = time.perf_counter() - starttime2
# 	print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
# 	starttime2 = time.perf_counter()

# totaltime = time.perf_counter() - starttime
# print("Compression comparisons done! Total execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

# #Save results to datafile for further processing
# data = asarray(results)
# save('original_vs_compressions.npy', data)
# print("Data saved!")


for y in [1,3,5,7,9]:
	print("Running the entire thing again but with reference image compression level " + str(y))
	#do main task, start by comparing original front to original side
	starttime = time.perf_counter()
	results = []

	print("Running compression with original comparison...")
	resultsarray = np.zeros((30,30))
	for i in range(1,31):
		for j in range(1,31):
			#print(str(i) + " " + str(j), end = "\r")
			resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "compression/compression_"+str(y)+"/"+str(i)+"/5.jpg", img2_path = "original/"+str(j)+"/1.jpg", model=model)['distance']
	results.append(resultsarray)

	print("Compression with original comparison done!")
	totaltime = time.perf_counter() - starttime
	print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))



	#Now compare original front to compressed side
	print("Running compression level " + str(y) + " vs compression comparison...")
	starttime2 = time.perf_counter()
	for h in [1,3,5,7,9]:
		resultsarray = np.zeros((30,30))
		for i in range(1,31):
			for j in range(1,31):
				#print(str(h) + " " + str(i) + " " + str(j), end = "\r")
				resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "compression/compression_"+str(y)+"/"+str(i)+"/5.jpg", img2_path = "compression/compression_"+str(h)+"/"+str(j)+"/1.jpg", model=model)['distance']
		results.append(resultsarray)
		print("Compression level " + str(h) + " done.")
		totaltime = time.perf_counter() - starttime2
		print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
		starttime2 = time.perf_counter()

	totaltime = time.perf_counter() - starttime
	print("Compression comparisons done! Total execution time for this round: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

	#Save results to datafile for further processing
	data = asarray(results)
	save('compression_'+str(y)+'_vs_compressions.npy', data)
	print("Data saved!")

print("Shutting down script.")