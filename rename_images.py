import glob
import cv2
import os 


files = ["file1", "file2", "file3"]
for file in files:
	images = glob.glob("./initial_file/"+file+"/*.*")
	for i in range(len(images)):
		t = "image"+str(i)+".jpg"
		image = cv2.imread(images[i])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		cv2.imwrite("./final_file/"+file+"/{}".format(t), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
