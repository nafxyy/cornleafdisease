from flask import Flask, render_template, request
import os
from ultralytics import YOLO
import cv2
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'tesdeploy/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # <- Tambahan penting
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = YOLO("tesdeploy/weights/best.pt")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filename = f"{uuid.uuid4().hex}.jpg"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(path)

            # Run detection
            results = model(path)
            result_img = results[0].plot()

            # Ambil nama class yang terdeteksi
            classes_detected = set()
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                classes_detected.add(cls_name)

            # Simpan gambar hasil deteksi
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_path, result_img)

            return render_template(
                'index.html',
                uploaded_image=filename,
                result_image='result_' + filename,
                classes_detected=classes_detected
            )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
