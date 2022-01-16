import numpy as np
from numpy import asarray
from numpy import load
import matplotlib.pyplot as plt

data = load('original_vs_compressions.npy')
thresholds = [0,0.01,0.03,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
AUCs = []

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

plt.plot(AUCs)
plt.show()