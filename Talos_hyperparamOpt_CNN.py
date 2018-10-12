import glob
import cv2
import talos
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
from keras.layers import  Convolution2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten, BatchNormalization, Activation
from keras.activations import relu, softmax
from keras.preprocessing.image import img_to_array


def CNNmodel(x_train, y_train, x_val, y_val, params):

	model = Sequential()
	model.add(Convolution2D(filters=16, 
				kernel_size=(3,3), 
				padding="same", 
				activation=relu, 
				use_bias=True, 
				input_shape=x_train.shape[1:]))
	model.add(Dropout(rate=.5))
	model.add(Convolution2D(filters=16, 
				kernel_size=(3,3),
				padding="same",
				activation=relu))
	model.add(Dropout(rate=.5))
	model.add(MaxPooling2D(pool_size=(2,2), 
			       strides=2)) 
	model.add(BatchNormalization(axis=-1))  
	model.add(Convolution2D(filters=32, 
				kernel_size=(3, 3), 
				activation=relu, 
				use_bias=True))
	model.add(Convolution2D(filters=64, 
				kernel_size=(3, 3), 
				activation=relu, 
				use_bias=True))
	model.add(MaxPooling2D(pool_size=(2,2), 
				strides=2))
	model.add(Flatten())
	model.add(Dense(output_dim=128))
	model.add(Activation(activation=relu))
	model.add(Dense(output_dim=y_train.shape[1], 
			activation=softmax))

	METRICS = ["accuracy"]
	LOSS = categorical_crossentropy
	OPTIMIZER = Adam(params["lr"])
	
	model.compile(loss=LOSS,
	              optimizer=OPTIMIZER,
	              metrics=METRICS)

	BATCH_SIZE = params["batch_size"]
	EPOCHS = params["epochs"]

	history = model.fit(x_train, y_train, 
				validation_split=.2,
				batch_size=BATCH_SIZE,
				epochs=EPOCHS,
				verbose=1)

	return history, model


classes = ["class0", "class1", "class2"]
data = []
labels = []
i = -1
image_dims = (64, 64)

for class_ in classes:
	i+=1
	imagePaths = glob.glob("./train_data/"+class_+"/*.*")
	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (image_dims[1], image_dims[0]))
		image = img_to_array(image)
		data.append(image)
		labels.append(i)

x_train = np.asarray(data)
y_train = np.asarray(labels)
y_train = np_utils.to_categorical(y_train)

p = { "lr" : [0.001, 0.0001, 0.00001],
	"batch_size": [64, 128, 256, 512],
    "epochs": [16, 32],
    "weight_regulizer":[None],
    "emb_output_dims": [None],}

h = talos.Scan(x=x_train, 
               y=y_train, 
               model=CNNmodel,
               search_method="random",
               val_split=0.1,
               params=p,
               dataset_name="classes_",
               experiment_no="0",
               grid_downsample= 0.3,
               save_best_model=True)

report = talos.Reporting("classes_0.csv")
