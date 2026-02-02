from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import io
import base64

app = Flask(__name__)

class SimpleAstroDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32*64*64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32*64*64)
        x = self.fc1(x)
        return torch.softmax(x, dim=1)

model = SimpleAstroDetector()
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def detect_objects(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # ✅ ФИЛЬТР АСТРОФОТО: 60%+ пикселей должны быть тёмными
    dark_pixels = np.sum(img_cv < 30)
    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    dark_ratio = dark_pixels / (total_pixels * 3)  # *3 для RGB
    
    if dark_ratio < 0.6:
        return [], 0, "❌ НЕ АСТРОФОТО (слишком светлое! Загружай чёрное небо)"
    
    # ✅ ТОЛЬКО АСТРО: усиливаем звёзды
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30, 
                              param1=40, param2=25, minRadius=3, maxRadius=60)
    
    objects = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles[:10]:
            objects.append({
                'x': x, 'y': y, 'r': r, 
                'type': 'звезда' if r<15 else 'планета'
            })
    
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_t)
        anomaly_score = pred[0][1].item()
    
    return objects, anomaly_score, "✅ АСТРОФОТО ОБНАРУЖЕНО!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    image = Image.open(file).convert('RGB')
    
    objects, anomaly, status = detect_objects(image)
    
    result = {
        'objects': objects,
        'anomaly_score': anomaly,
        'total': len(objects),
        'status': status,
        'image_b64': base64.b64encode(io.BytesIO(image.tobytes()).read()).decode()
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
