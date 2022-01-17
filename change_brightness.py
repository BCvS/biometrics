from PIL import Image, ImageEnhance
import os	

current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/brightness'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

for i in [0.1,0.5,1.5,3,5]:
	parent_dir = "C:/Users/spijk/Documents/_assignment/biometrics_data/images/brightness"
	compressionpath = 'brightness_' + str(i)
	path = os.path.join(parent_dir, compressionpath)
	os.mkdir(path)
	parent_dir = path
	for j in range(1,31):
		directory = str(j)
		path = os.path.join(parent_dir, directory)
		if not os.path.exists(path):
			os.mkdir(path)


def change_brightness_with_given_strength(image_path, output_path, strength):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    img_output = enhancer.enhance(strength)
    img_output.save(output_path)
    #return Image.open(output_path)

if __name__ == '__main__':
	for h in [0.1,0.5,1.5,3,5]:
		for i in range(1,31):
			for j in range(1,10):
				change_brightness_with_given_strength('images/original/'+str(i)+'/'+str(j)+'.jpg', 'images/brightness/brightness_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)