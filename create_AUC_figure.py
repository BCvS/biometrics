import numpy as np
from numpy import asarray
from numpy import load
import matplotlib.pyplot as plt

thresholds = np.arange(0, 1, 0.01).tolist()
AUCgraphs =[]

AUCs = []

data = load('C:/Users/spijk/Documents/_assignment/biometrics_data/saved_comparison_data/DeepFace/compression/original_vs_compressions.npy')
for comparison in data:
	false_matches_upper = np.triu(comparison, 1)
	false_matches_lower = np.tril(comparison, -1)
	correct_matches = np.diagonal(comparison)

	tprs = []
	fprs = []

	for threshold in thresholds:
		tns_u = (false_matches_upper > threshold).sum()
		tns_l = (false_matches_lower > threshold).sum()
		tns = tns_u + tns_l
		fps = 870 - tns
		tps = (correct_matches < threshold).sum() 
		fns = 30 - tps

		tpr = tps/(tps+fns)
		fpr = fps/(fps+tns)
		
		tprs.append(tpr)
		fprs.append(fpr)

	area = np.trapz(tprs, x=fprs)
	AUCs.append(area)

temp = []
temp.append(AUCs[0])
AUCs.reverse()
AUCs.pop()
temp.extend(AUCs)
AUCs = temp
#print(AUCs)
AUCgraphs.append(AUCs)

for i in  [9,7,5,3,1]:
	AUCs = []
	data = load('C:/Users/spijk/Documents/_assignment/biometrics_data/saved_comparison_data/DeepFace/compression/compression_'+str(i)+'_vs_compressions.npy')
	for comparison in data:
		false_matches_upper = np.triu(comparison, 1)
		false_matches_lower = np.tril(comparison, -1)
		correct_matches = np.diagonal(comparison)

		tprs = []
		fprs = []

		for threshold in thresholds:
			tns_u = (false_matches_upper > threshold).sum()
			tns_l = (false_matches_lower > threshold).sum()
			tns = tns_u + tns_l
			fps = 870 - tns
			tps = (correct_matches < threshold).sum() 
			fns = 30 - tps

			tpr = tps/(tps+fns)
			fpr = fps/(fps+tns)
			
			tprs.append(tpr)
			fprs.append(fpr)

		area = np.trapz(tprs, x=fprs)
		AUCs.append(area)
	temp = []
	temp.append(AUCs[0])
	AUCs.reverse()
	AUCs.pop()
	temp.extend(AUCs)
	AUCs = temp
	#print(AUCs)
	AUCgraphs.append(AUCs)

fig, ax = plt.subplots()
plt.ylim([0,1])
#print(AUCgraphs)
for AUCgraph in AUCgraphs:
	ax.plot(AUCgraph)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_xticks([0,1,2,3,4,5],labels=['No compression','9','7','5','3','1'])
ax.legend(['None', '9','7','5','3','1'], title="Ref image \n compression", loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.xlabel("Compression level test picture")
plt.ylabel("AUC value")

plt.show()


# data = load('compression_1_vs_compressions.npy')
# for comparison in data:
# 	false_matches_upper = np.triu(comparison, 1)
# 	false_matches_lower = np.tril(comparison, -1)
# 	correct_matches = np.diagonal(comparison)

# 	tprs = []
# 	fprs = []

# 	for threshold in thresholds:
# 		tns_u = (false_matches_upper > threshold).sum()
# 		tns_l = (false_matches_lower > threshold).sum()
# 		tns = tns_u + tns_l
# 		fps = 870 - tns
# 		tps = (correct_matches < threshold).sum() 
# 		fns = 30 - tps

# 		tpr = tps/(tps+fns)
# 		fpr = fps/(fps+tns)
		
# 		tprs.append(tpr)
# 		fprs.append(fpr)

# 	plt.plot(fprs,tprs)
# 	plt.show()