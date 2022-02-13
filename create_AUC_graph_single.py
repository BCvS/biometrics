#Code for creating a single AUC graph. Very similar to 'create_AUC_graphs_figure', but for testing purposes.
import numpy as np
from numpy import asarray
from numpy import load
import matplotlib.pyplot as plt
import os

#Reference and test angle numbers. Please keep in mind that the chosen specific combination has to have been done by 'create_comparison_tables.py'
reference_angle_number = 5
test_angle_number = 5

current_dir = os.getcwd()
#Thresholds used to calculate ROC graphs. True Positive, False Positive, True Negative, and False Negative rates are calculated for every threshold. The more threshold values, the more accurate the ROC graphs.
thresholds = np.arange(0, 1, 0.01).tolist()
model = 'Deepface' #other options: 'VGG-Face2','VGG-Face', 'Dlib', 'Facenet', 'ArcFace', 'Facenet512', 'OpenFace', 'DeepID'
comparisontype = 'noise' #other options: 'compression', 'resolution', 'brightness'

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

AUCgraphs =[]
AUCs = []

#Load in data and calculate ROCs, then AUCs, then AUC graphs.
data = load(current_dir+'/saved_comparison_data/'+model+'/'+str(reference_angle_number)+'-'+str(test_angle_number)+'/'+comparisontype+'/original_vs_'+comparisontype+'s.npy')
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

#Brightness values range between 0.1 to 5. The neutral ('original') value for brightness is 1, but we loaded it in first. We would like the brightness values in the graphs
#to be in chronological order, therefore we take the 'brightness 1' values from the start of the list, and insert them back into index 2.
if comparisontype == 'brightness':
	AUCs.insert(2, AUCs.pop(0))
AUCgraphs.append(AUCs)

#Because the naming convention of save files includes 'original' (a slight mistake for this exact reason) we first had to load in the files with 'original' in the name, 
#and now we can loop through the rest of the options
for option in options:
	AUCs = []
	data = load(current_dir+'/saved_comparison_data/'+model+'/'+str(reference_angle_number)+'-'+str(test_angle_number)+'/'+comparisontype+'/'+comparisontype+'_'+str(option)+'_vs_'+comparisontype+'s.npy')
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

	if comparisontype == 'brightness':
		AUCs.insert(2, AUCs.pop(0))
	AUCgraphs.append(AUCs)

if comparisontype == 'brightness':
	AUCgraphs.insert(2, AUCgraphs.pop(0))
fig, ax = plt.subplots()
plt.ylim([0,1])

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