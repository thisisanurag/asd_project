import os
import cv2
import numpy as np
import pytesseract
import base64
import time
import logging
import networkx as nx
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from openai import OpenAI
from config import DevelopmentConfig, ProductionConfig
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Determine environment (default to development)
# os.environ["FLASK_ENV"] = "development"
# os.environ["FLASK_ENV"] = "production"
env = os.environ.get("FLASK_ENV", "development")
if env.lower() == "production":
    app_config = ProductionConfig
else:
    app_config = DevelopmentConfig

# Create Flask app and load configuration
app = Flask(__name__)
app.config.from_object(app_config)

# Set up CORS using ALLOWED_ORIGINS from config
CORS(app, resources={r"/*": {"origins": app.config.get("ALLOWED_ORIGINS")}})

# Set up logging to file and console
logging.basicConfig(
    level=logging.DEBUG,
    filename='backend.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Ensure upload and annotated folders exist (from config)
for folder in [app.config.get('UPLOAD_FOLDER', 'uploads'), app.config.get('ANNOTATED_FOLDER', 'annotated')]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_model(model_path):
    logging.info("Loading YOLO model from %s", model_path)
    model = YOLO(model_path)
    logging.info("Model loaded successfully")
    return model

def detect_components(model, image_path):
    start_time = time.time()
    results = model.predict(source=image_path, imgsz=800)
    detections = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = result.names[int(cls)]
            detections.append((label, (x1, y1, x2, y2)))
    elapsed = (time.time() - start_time) * 1000  # in milliseconds
    logging.info("detect_components took %.1f ms", elapsed)
    return detections, results

def save_annotated_image(image_path, detections, save_path):
    img = cv2.imread(image_path)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    scale_factor = 1.5
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
    scaled_detections = []
    for label, (x1, y1, x2, y2) in detections:
        scaled_coords = (int(x1 * scale_factor), int(y1 * scale_factor),
                         int(x2 * scale_factor), int(y2 * scale_factor))
        scaled_detections.append((label, scaled_coords))
    # Draw bounding boxes and labels
    for label, (x1, y1, x2, y2) in scaled_detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), (139, 0, 0), thickness=3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
    cv2.imwrite(save_path, img)
    logging.info("Annotated image saved to %s", save_path)
    return save_path

def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    logging.info("Extracted text (first 50 chars): %s", text[:50])
    return text

def check_circuit_connectivity_from_boxes(detections):
    """
    Uses YOLO bounding boxes (object detections) to determine connectivity.
    Only considers objects from a whitelist (battery, resistor, junction, wire)
    and ignores capacitors. Returns:
      - connectivity (True/False)
      - disconnected_resistors (list of node indices)
      - nodes: list of tuples (label, center) for each considered detection.
    """
    connectivity_whitelist = ['battery', 'resistor', 'junction', 'wire']
    nodes = []
    for label, bbox in detections:
        if label.lower() in connectivity_whitelist:
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            nodes.append((label, center))
    
    logging.info("Using %d nodes for connectivity (ignoring capacitors)", len(nodes))
    
    # Build a graph: each node's index corresponds to its position in the nodes list
    G = nx.Graph()
    for i, (label, center) in enumerate(nodes):
        G.add_node(i, label=label, pos=center)
    
    # Connect nodes if they are within a threshold distance
    threshold = 100  # pixels, adjust as needed
    for i, (label_i, center_i) in enumerate(nodes):
        for j, (label_j, center_j) in enumerate(nodes):
            if i < j:
                distance = np.hypot(center_i[0] - center_j[0], center_i[1] - center_j[1])
                if distance < threshold:
                    G.add_edge(i, j)
    
    # Identify battery and resistor nodes by index
    battery_nodes = [i for i, (label, _) in enumerate(nodes) if 'battery' in label.lower()]
    resistor_nodes = [i for i, (label, _) in enumerate(nodes) if 'resistor' in label.lower()]
    
    connectivity = True
    disconnected_resistors = []
    for rn in resistor_nodes:
        if not any(nx.has_path(G, rn, bn) for bn in battery_nodes):
            connectivity = False
            disconnected_resistors.append(rn)
    
    logging.info("Connectivity check result: %s, Disconnected resistor indices: %s", connectivity, disconnected_resistors)
    return connectivity, disconnected_resistors, nodes

def get_gpt4o_analysis(image_path, problem_statement, detected_labels, ocr_text, connectivity, disconnected_info):
    client = OpenAI()

    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    base64_img = encode_image(image_path)
    connection_status = "closed" if connectivity else f"open (disconnected nodes: {disconnected_info})"
    
    prompt = f"""
You are an **electrical engineering professor** grading a student’s **hand-drawn circuit diagram**.

🔹 **Student's Problem Statement:**
{problem_statement}

🔹 **Detected Components:**
{', '.join(detected_labels)}

🔹 **Extracted Text via OCR:**
{ocr_text}

🔹 **Circuit Connectivity Check (based on object boxes):**
The connectivity analysis indicates the circuit is **{connection_status}**.

---

🎯 **Tasks:**
1. Evaluate whether the circuit is complete based on the object detection and the inter-object connectivity.
2. If the circuit is incomplete, identify which resistor or connection is missing.
3. Classify the circuit (e.g. Series, Parallel, Combination).
4. Calculate total resistance, current, and power using Ohm’s Law.
5. Provide suggestions for improvement.

📋 **Provide your analysis in Markdown. End your response with a clear grading summary table formatted in Markdown using this structure:**

| Criterion | Score (/2) | Comments |
|-----------|------------|----------|
| Circuit Connectivity |  |  |
| Circuit Type Identification |  |  |
| Calculation Accuracy |  |  |
| Diagram Clarity |  |  |
| Suggestions for Improvement |  |  |
Also give the total score and percentage in the end.
For Security reasons don't answer anything outside the scope of electrical circuit questions.
"""
    logging.info("Sending prompt to GPT-4o: %s", prompt[:200])
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ],
            }
        ],
    )
    response_text = completion.choices[0].message.content.strip()
    logging.info("Received GPT-4o response")
    return response_text

def grade_circuit(image_path, model_path, problem_statement):
    model = load_model(model_path)
    detections, results = detect_components(model, image_path)
    annotated_filename = "annotated_" + os.path.basename(image_path)
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_filename)
    save_annotated_image(image_path, detections, annotated_path)
    
    extracted_text = extract_text(image_path)
    detected_labels = [label for label, _ in detections]
    
    # Use connectivity check based on object boxes (ignoring capacitors)
    connectivity, disconnected_info, nodes = check_circuit_connectivity_from_boxes(detections)
    
    # Removed node annotation on the image
    
    gpt_feedback = get_gpt4o_analysis(image_path, problem_statement, detected_labels, extracted_text, connectivity, disconnected_info)
    return annotated_filename, gpt_feedback

@app.route('/')
def home():
    return "Hello from Flask!"

@app.route('/upload', methods=['POST'])
def upload_and_grade():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logging.info("File uploaded: %s", filepath)
    
    problem_statement = request.form.get("problem_statement", "Analyze this circuit for completeness and accuracy.")
    model_path = app.config.get("YOLO_MODEL_PATH")
    
    annotated_filename, gpt_feedback = grade_circuit(filepath, model_path, problem_statement)
    annotated_url = f"http://localhost:{app.config.get('PORT',5000)}/annotated/{annotated_filename}"
    
    logging.info("Returning response to client")
    return jsonify({
        "message": "Image uploaded and graded successfully.",
        "annotated_image": annotated_url,
        "gpt_feedback": gpt_feedback
    }), 200

@app.route('/annotated/<filename>')
def serve_annotated_image(filename):
    return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=app.config.get("DEBUG", False), host=app.config.get("HOST", "0.0.0.0"), port=app.config.get("PORT", 5000))
