from PIL import Image
import os

current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/resolution'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

for i in [1024, 512, 256, 128, 64]:
	parent_dir = req_dir
	compressionpath = 'resolution_' + str(i)
	path = os.path.join(parent_dir, compressionpath)
	if not os.path.exists(path):
		os.mkdir(path)
	parent_dir = path
	for j in range(1,31):
		directory = str(j)
		path = os.path.join(parent_dir, directory)
		if not os.path.exists(path):
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
		for i in range(1,31):
			for j in range(1,10):
				change_resolution('images/original/'+str(i)+'/'+str(j)+'.jpg', 'images/resolution/resolution_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)