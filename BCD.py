import numpy as np
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# 1. Load model
model = load_model('brain_cancer_model.h5')

# 2. Define classes exactly as per Colab training
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    img_file = request.files['file']

    if img_file.filename == '':
        return "No selected file"

    if img_file:
        # Define the path to save the uploaded image
        path = os.path.join("static", "temp_img.jpg")
        img_file.save(path)

        # 3. Load the image at 150x150 (Matches Colab Training)
        img = image.load_img(path, target_size=(150, 150))

        # 4. Convert to Array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)



        # 5. Predict
        preds = model.predict(x)
        print(f"Raw Probabilities: {preds}")  # Check your PyCharm terminal for this!

        result_index = np.argmax(preds)
        confidence = np.max(preds) * 100
        result_text = f"{classes[result_index]} ({confidence:.2f}%)"

        return render_template('index.html', prediction=result_text, img_path=path)


if __name__ == '__main__':
    app.run(debug=True)