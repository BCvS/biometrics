import numpy as np
from numpy import asarray
from numpy import load
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
thresholds = np.arange(0, 1, 0.01).tolist()
models = ['Deepface', 'VGG-Face', 'Facenet', 'OpenFace', 'Dlib'] #'Facenet512', 'ArcFace'
comparisontype = 'noise'

options = []
labels = []
if (comparisontype == 'resolution'):
	options = [1024, 512, 256, 128, 64]
	labels = ['Original', '1024', '512', '256', '128', '64']
elif (comparisontype == 'compression'):
	options = [9,7,5,3,1]
	labels = ['Original', '9', '7', '5', '3', '1']
elif (comparisontype == 'brightness'):
	options = [0.1,0.5,1.5,3,5]
	labels = ['0.1','0.5','Original','1.5','3','5']
elif (comparisontype == 'noise'):
	options = [0.1, 0.3, 0.5, 0.7, 1];
	labels = ['Original', '0.1', '0.3', '0.5', '0.7', '1']

for model in models:
	AUCgraphs =[]
	AUCs = []

	data = load(current_dir + '/saved_comparison_data/' + model + '/'+comparisontype+'/original_vs_'+comparisontype+'s.npy')
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

	# temp = []
	# temp.append(AUCs[0])
	# AUCs.reverse()
	# AUCs.pop()
	# temp.extend(AUCs)
	# AUCs = temp
	#print(AUCs)
	if comparisontype == 'brightness':
		AUCs.insert(2, AUCs.pop(0))
	AUCgraphs.append(AUCs)

	for i in options:
		AUCs = []
		data = load(current_dir+'/saved_comparison_data/' + model + '/'+comparisontype+'/'+comparisontype+'_'+str(i)+'_vs_'+comparisontype+'s.npy')
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
		# temp = []
		# temp.append(AUCs[0])
		# AUCs.reverse()
		# AUCs.pop()
		# temp.extend(AUCs)
		# AUCs = temp
		#print(AUCs)
		if comparisontype == 'brightness':
			AUCs.insert(2, AUCs.pop(0))
		AUCgraphs.append(AUCs)

	if comparisontype == 'brightness':
		AUCgraphs.insert(2, AUCgraphs.pop(0))
	fig, ax = plt.subplots()
	plt.ylim([0,1])
	#print(AUCgraphs)
	for AUCgraph in AUCgraphs:
		ax.plot(AUCgraph)

	box = ax.get_position()
	ax.set_title(model)
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.set_xticks([0,1,2,3,4,5],labels=labels)
	ax.legend(labels, title="Ref image\n"+comparisontype, loc='upper left', bbox_to_anchor=(1.0, 1.0))
	plt.xlabel("Test image " + comparisontype)
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