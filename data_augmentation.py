from keras.preprocessing.image import ImageDataGenerator


dataGeneration = ImageDataGenerator(rotation_range=5, horizontal_flip=True, 
				     width_shift_range=0.2, height_shift_range=0.2)
files = {"file1": 193, "file2": 400, "file3": 236}
for file in files.keys():
    i = 0
    for img in dataGeneration.flow_from_directory("./initial_data/"+brand, batch_size = 1,
        save_to_dir="./augmented_data/"+file, save_prefix="image", save_format='jpg'):
        i+=1
        if i >= files[file]:
            break
