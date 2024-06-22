# bone-fracture-7fylg_Yolov10
From dataset https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg  a model is obtained, based on yolov10, with that custom dataset, to indicate fractures in x-rays.



===
Installation:

Download all project datasets to a folder on disk.

Install yolov10  (if not yet installed) following the instructions given at:

https://blog.roboflow.com/yolov10-how-to-train/

which may be reduced to

 !pip install -q git+https://github.com/THU-MIG/yolov10.git

And download from https://github.com/THU-MIG/yolov10/releases the yolov10n.pt model. In case this operation causes problems, this file is attached with the rest of the project files.

Unzip the test.zip folder

Some zip decompressors duplicate the name of the folder to be decompressed; a folder that contains another folder with the same name, should only contain one. In these cases it will be enough to cut the innermost folder and copy it to the project folder.

===
Test:

It is executed:

EvaluateTESTFracture-7fylg_Yolov10.py

The x-rays are presented on the screen with a red box indicating the prediction . The console indicates the images in which no fracture has been detected and the errors asisigning fracture classes

The results are similar to those obtained by testing each image in the test file on the platform https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg/model/1. Is attached ComparingWithRoboflow.xlsx spreadsheet with the results 


===
 Training


The project comes with an optimized model: last28Hits190epoch.pt

To obtain this model, the following has been executed:

Download de file

 https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg

If you do not have a roboflow user key, you can obtain one at https://docs.roboflow.com/api-reference/authentication.

After downloading the dataset a folder named bone-fracture-7fylg_Yolov10 is created wich must be moved to the project folder( bone-fracture-7fylg_Yolov10)



Execute

TrainFracture-7fylg_Yolov10.py

This program has been adapted from https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85

It assumes that the project is located in the folder "C:/bone-fracture-7fylg_Yolov10", otherwise the assignment must be changed by modifying line 22

also uses the .yaml file:

data.yaml

In data.yaml the absolute addresses of the project appear assuming that it has been installed on disk C:, if it has another location these absolute addresses will have to be changed.



The best result has been found in the last.pt of 190 epoch, although when reaching the end, in 200 epoch, it seems that the values ​​of mAP50 and mAP50-95 are better. Which would indicate that the training has overfitting

===
References

https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg/model/1

https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85

https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch
