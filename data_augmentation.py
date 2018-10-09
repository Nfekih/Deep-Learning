import  glob
from keras.preprocessing.image import ImageDataGenerator

dataGeneration = ImageDataGenerator(rotation_range=5, horizontal_flip=True, 
				width_shift_range=0.2, height_shift_range=0.2)

brands = ["new balance" , "new converse", "new reebok",
		 "nike", "old converse", "old reebok" ]

totalIMG=1230 # total images for train and validation

for brand in brands:
	i=0
	images = glob.glob("./initialData/"+brand+"/"+brand+"/*.*")
	for img in dataGeneration.flow_from_directory("./initialData/"+brand,
							batch_size = 1, 
							save_to_dir="./augmentedData/"+brand,
							save_prefix="augIMG", 
							save_format='jpg'):
		i+=1
		if i+len(images) >= totalIMG:
	            break
