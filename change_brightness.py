#Code for generating images with brightness filters.
from PIL import Image, ImageEnhance
import os	

#Mininimum brightness is 0 (pitch black), there is no maximum brightness.
options = [0.1,0.5,1.5,3,5]
current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/brightness'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

#For saving the edited images, create a folder structure if it does not already exist.
for option in options:
	parent_dir = 'C:/Users/spijk/Documents/_assignment/biometrics_data/images/brightness'
	compressionpath = 'brightness_' + str(option)
	path = os.path.join(parent_dir, compressionpath)
	if not os.path.exists(req_dir):
		os.mkdir(path)
	parent_dir = path
	for face_number in range(1,31):
		directory = str(face_number)
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
	for option in options:
		for face_number in range(1,31):
			for face_angle_number in range(1,10):
				print('Changing brightness: brightness value '+str(option)+', face '+str(face_number)+', face angle ' + str(face_angle_number)+'	', end='\r')
				change_brightness_with_given_strength('images/original/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', 'images/brightness/brightness_'+str(option)+'/'+str(face_number)+'/'+str(face_angle_number)+'.jpg', option)

print('Brightness changes done, shutting down script.'+' '*30)