from flask import Flask, render_template, request, flash, send_from_directory,redirect, session, url_for
from werkzeug.utils import secure_filename
import os
import urllib.request
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image
import boto3


UPLOAD_FOLDER = 'C://Users//Dishant//Desktop//app//uploaded_image//'
IMAGE_SIZE = 300
model = load_model('C://Users//Dishant//Desktop//app//models//model-ep021-loss0.143-val_loss0.076.h5')

app = Flask(__name__)
app.secret_key='dishanttotade'

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('no file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('no image selected')
    else:
        filename  = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        flash('Image successfully uploaded and displayed below')
        img = Image.open(os.path.join(UPLOAD_FOLDER, filename))
        newfile = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        newfile.save(os.path.join(UPLOAD_FOLDER,filename))
        
        return render_template('index.html', filename = filename)

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(UPLOAD_FOLDER,
                               filename, as_attachment=True)

@app.route('/predict/<filename>')
def Cancer_prediction(filename):
    file = UPLOAD_FOLDER + filename
    img = image.load_img(file,target_size=(300,300))
    img = tf.keras.preprocessing.image.array_to_img(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    x = result.tolist()
    print("---------------------------------------------------")
    print(x)


    return render_template('index.html', filename=filename , data= x)





if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)