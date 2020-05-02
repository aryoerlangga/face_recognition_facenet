# Face Recognition using Facenet

This repository contains the demonstration on how to do facial recognition using facenet network. The code are heavily inspired by following [article](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)

### **How to use**
1. A pre-trained Keras Facenet model will be used to predict the embedding of an image. The model is provided by [Hiroki Taniai](https://github.com/nyoki-mtl/keras-facenet), and can be downloaded [here](https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_). Please save the downloaded model in [model](model) folder for code functionality.
2. Run the scripts [train.py](scripts/train.py) to train the model. However, the model is trained only using provided images inside [train folder](images/train). You can train using more images or label by adding them into the folder. Train folder structure are below.

```
face_recognition_facenet
|-- images
    |-- train
        |-- Steve Rogers
        |-- Thor
        |-- Tony Stark
        |-- [Another Person 1]
        |-- [Another Person 2]
        |-- [Another Person n]
```
3. After running the scripts, a face recognition model will be saved under `facereco.pkl` inside [model](model) folder, along with `target_encoder.pkl` as encoder
4. If you wish to predict an image using trained model, please put the image inside [predict_images](images/predict_images) folder. Use script [predict](scripts/predict.py). The prediction result will appear like following example.

![recognized the face!](images/result_example)

### **Norebook demo**
1. [MTCNN_demo](notebook_demo/MTCNN_demo.ipynb) </br>
Face detection using MTCNN
2. [train_demo](notebook_demo/train_demo.ipynb) </br>
Step by step on how to train the model
3. [predict_demo](notebook_demo/predict_demo.ipynb) </br>
Predict certain image using [predict.py](scripts/predict.py) script