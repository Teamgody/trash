
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# โหลดโมเดล (ตัวอย่าง dummy model หากไม่มีจริงให้ใส่ของคุณแทน)
model = tf.keras.models.load_model('model.h5')
class_names = ['organic', 'plastic', 'metal', 'glass', 'paper']

def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_image(filepath)
            image_path = url_for('static', filename='uploads/' + filename)

    return render_template('index.html', result=result, image=image_path)

if __name__ == '__main__':
    app.run(debug=True)
