# bone-fracture-7fylg_Yolov10
From dataset https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg a model is obtained, based on yolov10, with that custom dataset, to indicate fractures in x-rays. The project uses 5 cascade models, if one does not detect fracture it is passed to another

=== Installation: Download all project datasets to a folder on disk.

Install yolov10 (if not yet installed) following the instructions given at: https://blog.roboflow.com/yolov10-how-to-train/ which may be reduced to !pip install -q git+https://github. com/THU-MIG/yolov10.git

If you already have ultralytics installed, it would be advisable to upgrade ultralytics, unless you have applications based on yolov10 without updating, which could be affected by the update.

For that you must have an upgraded version of ultralytics and the proper version of lap inside conda in the scripts directory of the user environment:

python pip-script.py install --no-cache-dir "lapx>=0.5.2"

upgrade ultralytics:

python pip-script.py install --upgrade ultralytics

And download from https://github.com/THU-MIG/yolov10/releases the yolov10n.pt and yolov10s.pt model. In case this operation causes problems, these files are attached with the rest of the project files.

Unzip the test.zip folder Some zip decompressors duplicate the name of the folder to be decompressed; a folder that contains another folder with the same name, should only contain one. In these cases it will be enough to cut the innermost folder and copy it to the project folder.

=== Test: It is executed:

EvaluateTESTFracture-7fylg_Yolov10.py

The x-rays are presented on the screen with a red box indicating the prediction . The console indicates the images in which no fracture has been detected and the errors asigning fracture classes The results have been obtained using only one model.

In case of using several models: if one does not detect a fracture, another is tested and so on. The results are better at the risk of making false detections.

The results are obtained by executing:

EvaluateTESTFracture-7fylg_SeveralModels_Yolov10.py

The results are similar, even better, than those obtained by testing each image in the test file on the platform https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg/model/1.

Is attached ComparingWithRoboflow.xlsx spreadsheet with the results

=== Training

The project comes with an optimized model: last20epoch25hits.pt

To obtain this model, the following has been executed: Download de file https://universe.roboflow.com/roboflow- 100/bone-fracture-7fylg If you do not have a roboflow user key, you can obtain one at https://docs.roboflow.com/api-reference/authentication.

After downloading the dataset a folder named bone-fracture-2 is created which must be moved to the project folder( bone-fracture-7fylg_Yolov10)

Execute TrainFracture-7fylg_Yolov10.py

This program has been adapted from https://medium.com/@ huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85 It assumes that the project is located in the folder "C:/bone -fracture-7fylg_Yolov10", otherwise the assignment must be changed by modifying line 22 The parameter multi_scale has been changed to true.

also uses the .yaml file:

data.yaml

In data.yaml the absolute addresses of the project appear assuming that it has been installed on disk C:, if it has another location these absolute addresses will have to be changed.

Conclusions:

Instead  of performing a long training process, it has been preferred to use a short training process obtaining different models (some with yolov10s and others with yolov10n, see instructions 28 and 29 of TrainFracture-7fylg_Yolov10.py) and using the models in cascade so that If one did not detect fractures, the next one was passed on to see if he was able to detect them.

=== References

https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg/model/1

https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85 

https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch

https://github.com/ablanco1950/PointOutWristPositiveFracture_on_xray
