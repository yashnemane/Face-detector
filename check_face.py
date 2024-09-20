from flask import Flask, request, jsonify
from flask_cors import CORS
from mtcnn import MTCNN
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

detector = MTCNN()


@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files['image']
        image_data = np.fromstring(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        if len(faces) > 0:
            return jsonify({"message": "Face is present","isFacePresent": "true"}), 200
        else:
            return jsonify({"message": "No face detected","isFacePresent": "false"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)