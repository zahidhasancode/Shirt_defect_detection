import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = load_model('new1_shirt_defect_model.h5')
model.make_predict_function()

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Check if the file is allowed
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('index.html', error='Invalid file format')

        # Save the uploaded file temporarily
        uploaded_file_path = 'temp.jpg'
        file.save(uploaded_file_path)

        # Preprocess the uploaded image
        img = preprocess_image(uploaded_file_path)
        

       # Make a prediction
        prediction = model.predict(img)
    

        # Determine the result
        if prediction[0][0] > 0.5:
            result = 'Defective Shirt'
        else:
            result = 'Good Shirt'


        # Remove the temporary file
        os.remove(uploaded_file_path)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
