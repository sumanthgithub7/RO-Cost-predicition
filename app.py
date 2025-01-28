from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import torch
import pandas as pd
from torch import nn
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session
CORS(app)

# Configure paths
app.template_folder = 'templates'
app.static_folder = 'static'

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load preprocessor
preprocessor_path = r'C:\Users\suman\Desktop\Mini Project\model\preprocessor.pkl'
try:
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Preprocessor file not found at {preprocessor_path}")
    preprocessor = None

# Load PyTorch model
model_path = r'C:\Users\suman\Desktop\Mini Project\model\model.pth'
try:
    class FeedforwardNN(nn.Module):
        def __init__(self, input_size):
            super(FeedforwardNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.dropout1 = nn.Dropout(p=0.5)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(p=0.5)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 64)
            self.dropout3 = nn.Dropout(p=0.5)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(64, 1)

        def forward(self, x):
            x = self.dropout1(self.relu1(self.fc1(x)))
            x = self.dropout2(self.relu2(self.fc2(x)))
            x = self.dropout3(self.relu3(self.fc3(x)))
            x = self.fc4(x)
            return x

    model = FeedforwardNN(input_size=15)
    model.load_state_dict(torch.load(model_path))
    model.eval()
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/results')
def results():
    if 'prediction_result' not in session:
        return redirect(url_for('input_page'))
    return render_template('results.html', prediction=session['prediction_result'])

@app.route('/carbon')
def carbon_page():
    return render_template('carbon.html')

@app.route('/carbon_output')
def carbon_output():
    return render_template('carbon_out.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/water-quality')
def water_quality():
    return render_template('water_quality.html')

@app.route('/calculate-water-quality', methods=['POST'])
def calculate_water_quality():
    data = request.get_json()
    raw_salinity = float(data['rawSalinity'])
    plant_type = data['plantType']
    plant_location = data['plantLocation']
    plant_capacity = float(data['plantCapacity'])
    
    # Calculate TDS based on plant type efficiency
    efficiency = {
        'RO': 0.95,  # Reverse Osmosis has highest efficiency
        'MSF': 0.90, # Multi-Stage Flash
        'MED': 0.85, # Multi-Effect Distillation
        'EDR': 0.80  # Electrodialysis Reversal
    }
    
    tds_reduction = efficiency.get(plant_type, 0.85)
    predicted_tds = raw_salinity * (1 - tds_reduction)
    
    # Calculate Water Quality Index (WQI)
    if predicted_tds <= 500:
        wqi = 100
    elif predicted_tds <= 1000:
        wqi = 80
    elif predicted_tds <= 2000:
        wqi = 60
    elif predicted_tds <= 3000:
        wqi = 40
    else:
        wqi = 20
    
    # Determine water suitability
    suitability = {
        'drinking': predicted_tds <= 500,  # WHO guideline for drinking water
        'agriculture': predicted_tds <= 2000  # General threshold for agricultural use
    }
    
    return jsonify({
        'predictedTDS': round(predicted_tds, 2),
        'wqi': wqi,
        'suitability': suitability
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500

    try:
        data = request.get_json()
        input_data = pd.DataFrame([{
            'Plant Location': data['location'],
            'Plant Capacity (m³/day)': float(data['capacity']),
            'Project Award Year': int(data['year']),
            'Raw Water Salinity (mg/L)': float(data['salinity']),
            'Plant Type': data['plantType'],
            'Project Financing Type': data['financing'],
        }])

        # Apply preprocessing
        input_data = preprocessor.transform(input_data)

        # Convert to PyTorch tensor
        tensor_input = torch.tensor(input_data, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = model(tensor_input)
            predicted_cost = float(prediction.item())
            
            # Scale prediction if needed
            predicted_cost = abs(predicted_cost)
            if predicted_cost < 1000:
                predicted_cost *= 1_000_000

        # Store in session
        session['prediction_result'] = {
            'predicted_cost': predicted_cost,
            'input_data': {
                'location': data['location'],
                'capacity': data['capacity'],
                'year': data['year'],
                'salinity': data['salinity'],
                'plant_type': data['plantType'],
                'financing': data['financing']
            }
        }

        return jsonify({'redirect': url_for('results')})

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'Error processing request'}), 400

if __name__ == '__main__':
    app.run(debug=True)