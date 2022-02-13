#Code for creating the AUC graphs overview, after the required comparison tables have been calculated and saved.
import numpy as np
from numpy import asarray
from numpy import load
import matplotlib.pyplot as plt
import os

#Reference and test angle numbers. Please keep in mind that the chosen specific combination has to have been done by 'create_comparison_tables.py' 
#By default, combinations 5-1 and 5-5 have been done.
reference_angle_number = 5
test_angle_number = 5

current_dir = os.getcwd()
#Thresholds used to calculate ROC graphs. True Positive, False Positive, True Negative, and False Negative rates are calculated for every threshold. The more threshold values, the more accurate the ROC graphs.
thresholds = np.arange(0, 1, 0.01).tolist()
#Models to include. The default set below has been ordered by accuracy from high to low.
models = ['VGG-Face2','VGG-Face', 'Dlib', 'Facenet', 'ArcFace', 'Facenet512', 'OpenFace', 'DeepID', 'Deepface']
comparisontypes = ['compression', 'resolution', 'brightness', 'noise']

options = []
labels = []
fig, axs = plt.subplots(len(models),4, figsize=(12,15.5))

for comparisoncounter, comparisontype in enumerate(comparisontypes):
	if (comparisontype == 'resolution'):
		options = [1024, 512, 256, 128, 64]
		labels = ['2048', '1024', '512', '256', '128', '64']
	elif (comparisontype == 'compression'):
		options = [9,7,5,3,1]
		labels = ['100', '9', '7', '5', '3', '1']
	elif (comparisontype == 'brightness'):
		options = [0.1,0.5,1.5,3,5]
		labels = ['0.1','0.5','1','1.5','3','5']
	elif (comparisontype == 'noise'):
		options = [0.1, 0.3, 0.5, 0.7, 1];
		labels = ['0', '0.1', '0.3', '0.5', '0.7', '1']

	for modelcounter, model in enumerate(models):
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
		ax = axs[modelcounter,comparisoncounter]
		ax.set(ylim=[0,1])

		for AUCgraph in AUCgraphs:
			ax.plot(AUCgraph)

		#Setting legend texts and positions.
		box = ax.get_position()
		if comparisoncounter == 0:
			ax.set_title(model, rotation=90, x=-2.1*box.width, y=box.height*5.9, verticalalignment='center', size='x-large', fontweight='bold')
			ax.set_position([box.x0, box.y0, box.width, box.height])
			ax.set_ylabel("AUC value", size='large')
			ax.set_yticks([0,0.25,0.5,0.75,1],labels=[0,0.25,0.5,0.75,1],size='large')
		else: 
			ax.set_yticks([0,0.25,0.5,0.75,1],labels=[])
		if (modelcounter == len(models)-1):
			ax.set_xticks([0,1,2,3,4,5],labels=labels,size='large')
			ax.set_xlabel("Test image " + comparisontype, size='large')
		else:
			ax.set_xticks([0,1,2,3,4,5],labels=[])
		if(modelcounter == 0 and comparisontype == 'resolution'):
			ax.legend(labels, title="Ref image "+comparisontype, title_fontsize='large', fontsize='large', loc='upper center', bbox_to_anchor=(0, 1.23, 1.0, 0.5), ncol=3, handleheight=0.1, handlelength=1, labelspacing = 0.15, columnspacing=1.5, mode='expand')
		elif(modelcounter == 0 and comparisontype == 'brightness'):
			ax.legend(labels, title="Ref image "+comparisontype, title_fontsize='large', fontsize='large', loc='upper center', bbox_to_anchor=(0, 1.23, 1.0, 0.5), ncol=3, handleheight=0.1, handlelength=1.65, labelspacing = 0.15, columnspacing=1.5, mode='expand')
		elif modelcounter == 0:
			ax.legend(labels, title="Ref image "+comparisontype, title_fontsize='large', fontsize='large', loc='upper center', bbox_to_anchor=(0, 1.23, 1.0, 0.5), ncol=3, handleheight=0.1, handlelength=1.8, labelspacing = 0.15, columnspacing=1.5, mode='expand')

		ax.grid()

fig.suptitle('Reference face angle '+str(reference_angle_number)+' vs test face angle '+str(test_angle_number), y=0.995, fontsize='xx-large')
fig.tight_layout(rect=(0.015,0,1,1), h_pad=0.25)
fig.savefig(current_dir+'/_final_images/png/AUC-graphs_for_reference_angle_'+str(reference_angle_number)+'_and_test_angle_'+str(test_angle_number)+'.png')
fig.savefig(current_dir+'/_final_images/svg/AUC-graphs_for_reference_angle_'+str(reference_angle_number)+'_and_test_angle_'+str(test_angle_number)+'.svg')
print("Done! AUC graphs figure saved in _final_images.")
#plt.show()