import numpy as np
from numpy import asarray, load
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Arrow
from matplotlib.cm import ScalarMappable
import math
import os
import seaborn as sns; sns.set_theme();

comparisontype = 'resolution' #Choose the comparison type: 'brightness', 'noise', 'compression' or 'resolution'

brightness_options = [0.1,0.5,1.5,3,5]
compression_options = [9,7,5,3,1]
noise_options = [0.1, 0.3, 0.5, 0.7, 1]
resolution_options = [1024, 512, 256, 128, 64]

#Functional variables
current_dir = os.getcwd()
model = 'VGG-Face2'
thresholds = np.arange(0, 1, 0.0001).tolist()
options = []
labels = []

#Formatting variables
#For the colormap, the following function can also be used:
#sns.cubehelix_palette(start=0, rot=0.4, as_cmap=True)
#This requires editing the 'savefig' function at the end of the code, as 'color_palette' is used there.
color_palette = 'vlag'
colormap = sns.color_palette(color_palette, as_cmap=True)
lowest_AUC=0.45
left_offset = 0.032
right_offset = 0.97

subfig1_hr = 1.8
subfig2_hr = 10
subfig3_hr = 3.4
total_height_ratio = subfig1_hr + subfig2_hr + subfig3_hr
wording = ' level'
if (comparisontype=='brightness'):
	wording = ' (\u03B1 value)'
if (comparisontype=='noise'):
	wording = ' (\u03C3 value)'
if (comparisontype=='compression'):
	wording = ' (image quality)'
if (comparisontype=='resolution'):
	wording = ' (pixel width)'

#Create figure and split into three stacked subfigures. One for the example images, one for the AUC grid, one for a magnified AUC table.
fig = plt.figure(constrained_layout=False, figsize=(9.5,12.6))
subfigs = fig.subfigures(3, 1, height_ratios=[subfig1_hr, subfig2_hr, subfig3_hr])

axs0 = subfigs[0].subplots(1, 9)
subfigs[0].subplots_adjust(bottom=-0.2, top=0.75, wspace=0.1, hspace=0.1, left=left_offset, right=right_offset-0.1*(right_offset-left_offset))
subfigs[0].supxlabel('Reference face angle', y=0.58)
subfigs[0].suptitle(comparisontype.title() + wording+ ' and face angle comparison', fontsize='xx-large')

axs1 = subfigs[1].subplots(9, 9)
#subfigs[1].set_facecolor('0.9')
subfigs[1].supylabel('Test face angle', x=0.005)
subfigs[1].subplots_adjust(bottom=0.03, top=0.97, wspace=0.1, hspace=0.1, left=left_offset, right=right_offset)

axs2 = subfigs[2].subplots(1, 2)
subfigs[2].supxlabel('Reference '+comparisontype+wording, y=0)
subfigs[2].supylabel('Test '+comparisontype+'\n'+wording, x=0.09, y=0.55)
subfigs[2].subplots_adjust(left=0.18,right=0.82, bottom=0.2, top=0.92, wspace=0.82)

#Fill in labels and options based on comparison type choice
if (comparisontype == 'resolution'):
	options = resolution_options
	labels = ['2048'] + list(map(str, options))
elif (comparisontype == 'compression'):
	options = compression_options
	labels = ['95'] + list(map(str, options))
elif (comparisontype == 'brightness'):
	options = brightness_options
	labels = list(map(str, options))
	labels.insert(2, '1')
elif (comparisontype == 'noise'):
	options = noise_options
	labels = ['0'] + list(map(str, options))

#Load in example images for every face angle
for i in range(0,9):
	ax = axs0[i]
	img = mpimg.imread(current_dir+'/faces/original/1/' + str(i+1) + '.jpg')
	ax.imshow(img)
	ax.set_xticks([])
	ax.set_yticks([])

#Main part of the code: for every face angle combination, calculate an AUC table for every value combination of the current comparison type
for i in range(1,10):
	for j in range(1,10):
		AUCgraphs =[]
		AUCs = []
		data = load(current_dir+'/saved_comparison_data/'+model+'/'+str(i)+'-'+str(j)+'/'+comparisontype+'/original_vs_'+comparisontype+'s.npy')
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
			data = load(current_dir+'/saved_comparison_data/'+model+'/'+str(i)+'-'+str(j)+'/'+comparisontype+'/'+comparisontype+'_'+str(option)+'_vs_'+comparisontype+'s.npy')
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

		#Draw the heatmaps.
		ax = axs1[8-(j-1),i-1]
		heatmap = sns.heatmap(AUCgraphs, vmin=lowest_AUC, vmax=1, cmap=colormap, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
		heatmap.invert_yaxis()

		#This part is for setting various legend and label values at the edges of the heatplot grid.
		if(j==9):
			ax.set_title(i, y=0.97)
		if(i==9):
			if(j==1):	
				ax.set_ylabel(j, rotation=0, labelpad=12, y=0.56)
			else:
				ax.set_ylabel(j, rotation=0, labelpad=8.5, y=0.56)
			ax.yaxis.set_label_position("right")

		if(i==1 and j==1):
			heatmap2 = sns.heatmap(AUCgraphs, vmin=lowest_AUC, vmax=1, cmap=colormap, cbar=False, ax=axs2[0], xticklabels=labels, yticklabels=labels)
			heatmap2.invert_yaxis()
			heatmap2.set_yticklabels(heatmap2.get_yticklabels(), rotation = 0)
		if(i==9 and j==1):
			heatmap2 = sns.heatmap(AUCgraphs, vmin=lowest_AUC, vmax=1, cmap=colormap, cbar=False, ax=axs2[1], xticklabels=labels, yticklabels=labels)
			heatmap2.invert_yaxis()
			heatmap2.set_yticklabels(heatmap2.get_yticklabels(), rotation = 0)
		print('Progress: ' + str(i) + '-' + str(j), end="\r")

#This part is for drawing the two black rectangles and arrows, and inserting the colorbar. Because of the left and right offset and the inserted colorbar, relative proportions get complicated.
#Some black-box correction factors were used to get the boxes where they should be. This part could be made more accurate, but for the default figure size, it works.
#The location of the arrowheads are hardcoded, and need to be adjusted if the figure dimensions- or proportions are changed.
boxdistance = 0.008
cb_fraction = 0.05
cb_space_fraction = 0.05 #rough estimation of the starting space allocated by matplotlib for the colorbar 
correction_factor = 0.5 #no idea why, but this works
correction_factor_box2 = 1.2 #still no idea

ax = axs1[8,0]
x_start = ((ax.get_position().x0 - (ax.get_position().x0 - left_offset) * cb_space_fraction) * (1-(cb_fraction*correction_factor))) - boxdistance * (1-cb_space_fraction) * (1-cb_fraction) * correction_factor_box2
y_start = (ax.get_position().y0 * (subfig2_hr/total_height_ratio) + (subfig3_hr/total_height_ratio)) - boxdistance
x_length = ((ax.get_position().x1 - (ax.get_position().x1 - left_offset) * cb_space_fraction) * (1-(cb_fraction*correction_factor)) - x_start) + boxdistance * (1-cb_space_fraction) * (1-cb_fraction)
y_length = ((ax.get_position().y1 * (subfig2_hr/total_height_ratio) + (subfig3_hr/total_height_ratio)) - y_start) + boxdistance

subfigs[2].patches.extend([Rectangle((x_start,y_start),x_length,y_length, fill=False, color='black', lw=2, zorder=1000, transform=subfigs[2].transFigure)])
subfigs[2].patches.extend([Arrow(x_start+x_length,y_start,0.044,-0.025, color='black', width=0.01, transform=subfigs[2].transFigure)])

ax = axs1[8,8]
x_start = ((ax.get_position().x0 - (ax.get_position().x0 - left_offset) * cb_space_fraction) * (1-(cb_fraction))) - boxdistance * (1-cb_space_fraction + cb_fraction) * correction_factor_box2
y_start = (ax.get_position().y0 * (subfig2_hr/total_height_ratio) + (subfig3_hr/total_height_ratio)) - boxdistance
x_length = ((ax.get_position().x1 - (ax.get_position().x1 - left_offset) * cb_space_fraction) * (1-(cb_fraction)) - x_start) + boxdistance * (1-cb_space_fraction) * (1-cb_fraction) * correction_factor_box2
y_length = ((ax.get_position().y1 * (subfig2_hr/total_height_ratio) + (subfig3_hr/total_height_ratio)) - y_start) + boxdistance

subfigs[2].patches.extend([Rectangle((x_start,y_start),x_length,y_length, fill=False, color='black', lw=2, zorder=1000, transform=subfigs[2].transFigure)])
subfigs[2].patches.extend([Arrow(x_start+x_length*0.5,y_start-0.001,-0.012,-0.025, color='black', width=0.01, transform=subfigs[2].transFigure)])

norm = Normalize(vmin=lowest_AUC, vmax=1, clip=False,)
sm = ScalarMappable(norm=norm, cmap=colormap)

plt.rcParams['axes.grid'] = False #gets rid of annoying deprecation warning
colorbar = subfigs[1].colorbar(sm, location='right', ax=axs1, shrink=0.9, fraction=cb_fraction, aspect=35, ticks=[round(x * 0.01, 2) for x in range(45, 105, 5)])
colorbar.ax.set_title('AUC', fontsize='large', y=1.01, weight='bold')

#Save figures as PNG and scalable SVG.
fig.savefig(current_dir+'/_final_images/png/'+comparisontype+'_face_angle_comparison_'+color_palette+'.png')
fig.savefig(current_dir+'/_final_images/svg/'+comparisontype+'_face_angle_comparison_'+color_palette+'.svg')
print("Done! AUC tables chart image saved in _final_images.")
#plt.show()