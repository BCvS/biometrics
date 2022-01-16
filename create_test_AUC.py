import numpy as np
from numpy import asarray
from numpy import load
import matplotlib.pyplot as plt

thresholds = np.arange(0, 1, 0.01).tolist()
AUCgraphs =[]

AUCs = []

data = load('compression_1_test.npy')
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

	plt.plot(fprs,tprs)
	plt.show()
	area = np.trapz(tprs, x=fprs)
	AUCs.append(area)

fig, ax = plt.subplots()
ax.plot(AUCs)
plt.ylim([0,1])
ax.set_xticks([0,1,2,3,4,5],labels=['No compression','9','7','5','3','1'])
plt.xlabel("Compression level test picture")
plt.ylabel("AUC value")

plt.show()


