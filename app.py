import json
import os
import requests

from torchvision import models
from flask import Flask, request, redirect, render_template
from commons import format_class_name
from inference import get_prediction

app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


# Predict by uploading file image
@app.route('/predict/', methods=['POST', 'GET'])
def predict_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name)
    return render_template('index.html')


# Predict by passing url link
@app.route('/predict_url')
def predict_url():
    url_link = request.args['url_link']
    url_image = requests.get(url_link)
    image_bytes = url_image.content
    class_id, class_name = get_prediction(image_bytes=image_bytes)
    class_name = format_class_name(class_name)
    return render_template('result.html', class_name=class_name)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))
