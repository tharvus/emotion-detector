# emotion-detector

## Description
A program which opens the user's camera and predicts the emotions of all the faces appearing in the video in real time. 3 emotions are predicted: Happy, Sad and Neutral

## Main concepts used
This project was built using tensorflow by leveraging transfer learning on VGG16. The model was trained using Kaggle's free GPU and the notebook is attached in the Model folder

## Dataset used
The dataset used was the FER-2013 dataset on kaggle. 
Link: https://www.kaggle.com/datasets/msambare/fer2013 

## Running the project
It is highly encourages to create and activate a virtual environment before performing following steps.

### Step 1: Downloading the model
Open the following link and download the model into a folder called "Model". 

https://drive.google.com/file/d/1jm3jsuEACozSz7pEy0y6vFx1sSoaaB9Z/view?usp=sharing 

If you wish to install the .h5 file in another directory, you may, but you will need to change the `PATH` variable in app.py to whichever relative path the model is located at.

### Step 2: Installing the required libraries
Run the following command

```
pip install -r requirements.txt
```

to install all the libraries required

### Step 3: Run the application and configure device
Go into the App folder by using the command `cd App` and type 

```
python app.py
```

This should open a camera.

In case it does not, it may be because you are using an external webcam, in which case, you need to edit the number in

```
video = cv2.VideoCapture(0) 
```
to whichever device number you are running. 

### Step 4: Quitting the application
The application should now be running and predicting your face. 

To quit, tap the screen where the camera is predicting your emotion and press the "q" key to stop the application.