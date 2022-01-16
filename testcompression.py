from deepface import DeepFace
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from numpy import asarray
from numpy import save


print("Running compression level 1 test")
print("Building model...")
model = DeepFace.build_model("DeepFace")
print("Model built!")
#do main task, start by comparing original front to original side
starttime = time.perf_counter()
results = []

print("Running compression with original comparison...")
resultsarray = np.zeros((30,30))
for i in range(1,31):
	for j in range(1,31):
		#print(str(i) + " " + str(j), end = "\r")
		resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "compression/compression_1/"+str(i)+"/5.jpg", img2_path = "original/"+str(j)+"/1.jpg", model=model)['distance']
results.append(resultsarray)

print("Compression with original comparison done!")
totaltime = time.perf_counter() - starttime
print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))



#Now compare original front to compressed side
starttime2 = time.perf_counter()
for h in [9,7,5,3,1]:
	resultsarray = np.zeros((30,30))
	for i in range(1,31):
		for j in range(1,31):
			#print(str(h) + " " + str(i) + " " + str(j), end = "\r")
			resultsarray[i-1,j-1] = DeepFace.verify(img1_path = "compression/compression_1/"+str(i)+"/5.jpg", img2_path = "compression/compression_"+str(h)+"/"+str(j)+"/1.jpg", model=model)['distance']
	results.append(resultsarray)
	print("Compression level " + str(h) + " done.")
	totaltime = time.perf_counter() - starttime2
	print("Execution time: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))
	starttime2 = time.perf_counter()

totaltime = time.perf_counter() - starttime
print("Compression comparisons done! Total execution time for this round: " + str(time.strftime('%H:%M:%S', time.gmtime(totaltime))))

#Save results to datafile for further processing
data = asarray(results)
save('compression_1_test.npy', data)
print("Data saved!")
