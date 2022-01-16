from PIL import Image
import os

for i in [1024, 512, 256, 128, 64]:
	parent_dir = "C:/Users/spijk/Documents/_assignment/biometrics_data/resolution"
	compressionpath = 'resolution_' + str(i)
	path = os.path.join(parent_dir, compressionpath)
	os.mkdir(path)
	parent_dir = path
	for j in range(1,31):
		directory = str(j)
		path = os.path.join(parent_dir, directory)
		os.mkdir(path)
print("Directories generated")

def change_resolution(image_path, output_path, width):
	basewidth = width
	img = Image.open(image_path)
	wpercent = (basewidth / float(img.size[0]))
	hsize = int((float(img.size[1]) * float(wpercent)))
	img = img.resize((basewidth, hsize), Image.ANTIALIAS)
	img.save(output_path)

if __name__ == '__main__':
	for h in [1024, 512, 256, 128, 64]:
		image_path_basis = "C:/Users/spijk/Documents/_assignment/biometrics_data/resolution"
		output_path_basis = os.path.join("C:/Users/spijk/Documents/_assignment/biometrics_data/resolution", "resolution_" + str(h))
		for i in range(1,31):
			for j in range(1,10):
				change_resolution('original/'+str(i)+'/'+str(j)+'.jpg', 'resolution/resolution_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)