import numpy as npp
import os
import sys
import re 
import glob
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import os

# Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# Define a flask app
app = Flask(__name__)
# Define model path
model_path = "D:/NLP/NLPCOMPLETE/vgg19.h5"

# Load model
model = load_model(model_path)
model.make_predict_function()

## load model  
model = load_model(model_path)
model.make_predict_function()    # Necessary
## preprocessing function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    ##Preprocessing the image
    x = image.img_to_array(img)
    ## x= np.true_divide(x, 255.0) # Normalize the image
    x = np.expand_dims(x, axis=0)
    x= preprocess_input(x)  # Preprocess the image for VGG19
    """" Be careful how your trained model deals with the input.
        otherwie it wonts make correct predictions"""
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        f = request.files['file']
        # Save the file to a temporary location
        basepath = os.path.dirname(__file__)  # Current directory
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        pred = model_predict(file_path, model)
        ## Here we make you     predictions
        pred_class=decode_predictions(pred, top=1)
        result = str(pred_class[0][0][1]) 
        # Return the result
        return result
    return None
        

if __name__=='__main__': 
    app.run(debug=True)

    