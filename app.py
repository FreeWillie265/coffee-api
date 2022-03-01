import io
import string
from unicodedata import name
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, jsonify, request, render_template
from PIL import Image

app = Flask(__name__)

# Modelling Task
model = models.resnext101_32x8d()
num_inftr = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_inftr, 128), 
                              nn.ReLU(), 
                              nn.Linear(128, 256), 
                              nn.ReLU(), 
                              nn.Linear(256, 512),  
                              nn.ReLU(),
                              nn.Linear(512, 1) , 
                              nn.Sigmoid())
model.load_state_dict(torch.load('./model.pkt'))
model.eval()

class_names = ['Healthy', 'Unhealthy']

def transform_image(image_bytes):
	my_transforms = transforms.Compose([
		transforms.Resize(256), #change to 224
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model.forward(tensor)
	_, prediction = torch.max(outputs, 1)
	return class_names[prediction]

diseases = {
    "Healthy": "This leaf is healthy",
    "Unhealthy": "This coffee leaf is unhealty"
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        image_bytes = file.read()
        prediction_name = get_prediction(image_bytes)
        return render_template('result.html',
            name=prediction_name.lower(),
            description=diseases[prediction_name])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
