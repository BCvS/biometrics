from PIL import Image
import os	

current_dir = os.getcwd()

req_dir = current_dir + '/images'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

req_dir = req_dir + '/compression'
if not os.path.exists(req_dir):
	os.mkdir(req_dir)

for i in [1,3,5,7,9]:
	parent_dir = req_dir
	compressionpath = 'compression_' + str(i)
	path = os.path.join(parent_dir, compressionpath)
	os.mkdir(path)
	parent_dir = path
	for j in range(1,31):
		directory = str(j)
		path = os.path.join(parent_dir, directory)
		if not os.path.exists(path):
			os.mkdir(path)


# parent_dir = "C:/Users/spijk/Documents/_assignment/biometrics_data/compression_5"
# for i in range(1,31):
# 	directory = str(i)
# 	path = os.path.join(parent_dir, directory)
# 	os.mkdir(path)

def compress_with_given_quality(image_path, output_path, quality=10):
    img = Image.open(image_path)
    img.save(output_path, quality=quality)
    return Image.open(output_path)

if __name__ == '__main__':
	for h in [1,3,5,7,9]:
		for i in range(1,31):
			for j in range(1,10):
				compress_with_given_quality('images/original/'+str(i)+'/'+str(j)+'.jpg', 'images/compression/compression_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)