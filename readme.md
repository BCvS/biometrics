# Biometrics CNN face recognition performance comparison using Serengil's deepface library

In this project, several face recognition algorithms were tested for accuracy at multiple face angles when images are distorted. 

## Process

30 faces at 9 different angles each were hand-picked from the PUT face image dataset (Andrzej Kasinski, Andrzej Florek, Adam Schmidt, 2008). 

Images were distorted by applying one of the following 4 effects at varying degrees:
- resolution reduction
- compression
- brightness filter (darker or lighter)
- noise

The algorithms used are:
- Face **extraction** algorithm: mtcnn
- Face **recognition** algorithms with serengil's [deepface](https://github.com/serengil/deepface) library: VGG-Face, Dlib, Facenet, ArcFace, Facenet512, OpenFace, DeepID, Deepface
- Additional face **recognition** algorithm: VGG-Face2 (technically not an algorithm but an image set, what was used in this project is the resnet50 algorithm trained on the VGG-Face2 image set)

To determine accuracy, ROCs (Receiver Operating Characteristic graphs) were calculated, and AUC (Area Under Curve) values were determined using those. AUC values were then visualised in graphs and colorized tables.

In short, the process can be summarized as: 
1. pick dataset
2. pick faces and face angles
3. apply the 4 distortions
4. extract faces using mtcnn
5. use face recognition algorithms on faces to calculate similarity scores
6. from similarity scores, calculate ROCs, then AUCs
7. visualize AUCs in plots and tables 
8. profit


# Usage
The github project comes with the entire process already done and the AUC figures generated and saved in the folder _final_images. If you want to do your own tests or use this project for your own goals, here is how to use it and run the scripts:

## Required libraries
The project is run using python 3.6. Libraries include [deepface](https://github.com/serengil/deepface), [matplotlib](https://matplotlib.org/stable/users/installing/index.html), [numpy](https://numpy.org/install/), [scikit-image](https://pypi.org/project/scikit-image/), [seaborn](https://seaborn.pydata.org/installing.html), [scipy](https://scipy.org/install/), [mtcnn](https://pypi.org/project/mtcnn/), [tensorflow](https://www.tensorflow.org/install/pip), and [keras](https://pypi.org/project/keras/). Getting tensorflow to work correctly and use your systems' GPU can be a bit of a hassle. Make sure to also install [CUDA and cuDNN](https://www.tensorflow.org/install/gpu).

## Scripts

First, the 30 original images with face angles from the PUT dataset are already present, and located in /images/original. 
1. To apply the distortions, run the following 4 scripts:
- compress.py
- change_brightness.py
- change_resolution.py
- noisify.py (this one takes the longest by far)

The images are then saved in the /images subfolders. 
If you wanted to play around with the endresults you could, for example, change the strength of the applied distortions. The currently used values are located at the top of the scripts.

2. Then to extract the faces, run extract_faces.py. This process takes some time, but saves a lot of it later on.

3. To determine and save the similarity scores, you have to run 2 scripts that roughly do the same thing. 
- First, you can change the the variables *models*, *reference_angle_number* and *test_angle_number* in the create_comparison_tables.py to your liking, then run the script. If you changed the distortion strengths before, make sure to change them here accordingly as well in at the top of the script. **The script itself is slow and can take up to a day to execute on a GTX 3090, if all *models* are included.** Ultimately, the deepface library is inefficient for a large-scale comparison project like this because of technicalities.
- The latter becomes clear once you run the second script, VGGFace2_AUC_comparison_tables_all_face-angles.py. Once again, make sure the distortion strengths are the same as in the previous step. This script does the same thing as the previous one, except for that it calculates similarity scores for a single model, namely VGG-Face2. Or rather, resnet50 trained of the VGG-Face2 dataset. It does not use the deepface library but builds the resnet50 model manually. This speeds up the process by such a large factor that we can easily calculate all 9 face angle comparisons for all 4 distortion types in a matter of minutes!

After running these two scripts, similarity scores should have been saved the saved_comparison_data folder. 

4. Finally, we want to calculate the AUC values and plot them in graphs and tables. You can view a single test graph by running create_AUC_graph_single.py. Otherwise:
- For the graphs, run create_AUC_graphs_figure.py. Make sure to first update the variables *models*, *reference_angle_number*, and *test_angle_number* and the distortion values from step 1 if you have changed any of them so far. This script creates a figure with graphs that show the accuracies of all models at all distortion values.
- For the tables, run create_AUC_tables_chart.py. For this one, choose the comparison type you want to view, and update distortion values if changed earlier. This script creates a single large table comprised of multiple tiny tables. The tables show the AUC values for the VGG-Face2 algorithm for every distortion value combination and every face angle combination. This is mainly possible thanks to the low excecution time of the VGGFace2_AUC_comparison_tables_all_face-angles.py script in the previous step.

**Done!** All the figures should now be saved in the _final_images folder. 

If you want to try some different colour palettes for the AUC table in the second script of step 4, I have included some colour palette example images in /images/test_colour_palettes.
