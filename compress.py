from PIL import Image

import os	

# for i in [1,3,5,7,9]:
# 	parent_dir = "C:/Users/spijk/Documents/_assignment/biometrics_data/compression"
# 	compressionpath = 'compression_' + str(i)
# 	path = os.path.join(parent_dir, compressionpath)
# 	os.mkdir(path)
# 	parent_dir = path
# 	for j in range(1,31):
# 		directory = str(j)
# 		path = os.path.join(parent_dir, directory)
# 		os.mkdir(path)


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
		image_path_basis = "C:/Users/spijk/Documents/_assignment/biometrics_data/original"
		output_path_basis = os.path.join("C:/Users/spijk/Documents/_assignment/biometrics_data", "compression_" + str(h))
		for i in range(1,31):
			for j in range(1,10):
				compress_with_given_quality('original/'+str(i)+'/'+str(j)+'.jpg', 'compression/compression_'+str(h)+'/'+str(i)+'/'+str(j)+'.jpg', h)