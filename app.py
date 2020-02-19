import json
import os
import requests

from PIL import Image
from torchvision import models
from flask import Flask, jsonify, request, redirect, render_template
from commons import format_class_name, url_loader
from inference import get_prediction

app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


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


@app.route('/url', defaults={'url_image': ''})
@app.route('/url=<url_image>', methods=['GET'])
def predict_url_image():
    # IMAGE_URL = "https://www.akc.org/wp-content/themes/akc/component-library/assets/img/welcome.jpg"
    url_image = request.args.get('url_image')  # read image URL as a request URL param
    # image = url_loader()
    # response = requests.get(image)
    img_bytes = url_image.read()
    class_id, class_name = get_prediction(image_bytes=img_bytes)
    class_name = format_class_name(class_name)
    return render_template('predict.html', class_name=class_name)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))
