from PIL import Image, ImageEnhance

import os	

# for i in [0.1,0.5,1.5,3,5]:
# 	parent_dir = "C:/Users/spijk/Documents/_assignment/biometrics_data/brightness"
# 	compressionpath = 'brightness_' + str(i)
# 	path = os.path.join(parent_dir, compressionpath)
# 	os.mkdir(path)
# 	parent_dir = path
# 	for j in range(1,31):
# 		directory = str(j)
# 		path = os.path.join(parent_dir, directory)
# 		os.mkdir(path)


def change_brightness_with_given_strength(image_path, output_path, strength):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    img_output = enhancer.enhance(strength)
    img_output.save(output_path)
    #return Image.open(output_path)

if __name__ == '__main__':
	for h in [0.1,0.5,1.5,3,5]:
		image_path_basis = "C:/Users/spijk/Documents/_assignment/biometrics_data/brightness"
		output_path_basis = os.path.join("C:/Users/spijk/Documents/_assignment/biometrics_data/brightness", "brightness_" + str(h))
		for i in range(1,31):
			for j in range(1,10):
				change_brightness_with_given_strength('original/'+str(i)+'/'+str(j)+'.jpg', 'brightness/brightness_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)