import numpy as np
from numpy import asarray
from numpy import load
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
thresholds = np.arange(0, 1, 0.01).tolist()
models = ['VGG-Face2','VGG-Face', 'Dlib', 'Facenet', 'ArcFace', 'Facenet512', 'OpenFace', 'DeepID', 'Deepface']
comparisontypes = ['compression', 'resolution', 'brightness', 'noise']

options = []
labels = []
fig, axs = plt.subplots(len(models),4, figsize=(12,15))

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
		ax = axs[modelcounter,comparisoncounter]
		ax.set(ylim=[0,1])
		#print(AUCgraphs)
		for AUCgraph in AUCgraphs:
			ax.plot(AUCgraph)

		box = ax.get_position()
		if comparisoncounter == 0:
			ax.set_title(model, rotation=90, x=-2.1*box.width, y=box.height*5.9, verticalalignment='center', size='x-large', fontweight='bold')
			ax.set_position([box.x0, box.y0, box.width * 1, box.height])
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

fig.tight_layout(rect=(0.015,0,1,1), h_pad=0.25)
fig.savefig('test.png')
#plt.show()