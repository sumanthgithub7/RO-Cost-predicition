# 🌊 RO-Based Desalination Cost Prediction

## 📌 Overview
This project uses a **multi-layer feedforward neural network** with backpropagation to predict the **capital cost** of Reverse Osmosis (RO) desalination plants. The model is trained on a dataset of **1806 RO plants** with capacities of at least **1000 m³/day**, utilizing six input parameters.

## ✨ Features
- 🔹 **Capital Cost Prediction:** Estimates the investment cost of new RO desalination plants.
- 🔹 **Carbon Footprint Calculator:** Assesses the environmental impact of desalination plants based on energy consumption and emissions.
- 🔹 **Water Quality Estimation:** Uses **TDS (Total Dissolved Solids)** and other parameters to determine water quality and assess its usability.
- 🔹 **Dynamic UI:** Interactive frontend showcasing all features with seamless navigation.

## 📊 Dataset
The model is trained on a dataset containing:
- 📍 **Plant Location** (categorical)
- ⚡ **Plant Capacity** (numerical)
- 📆 **Project Award Year** (numerical)
- 💧 **Raw Water Salinity** (numerical)
- 🏭 **Plant Type** (categorical)
- 💰 **Project Financing Type** (categorical)
- 💵 **Capital Cost** (output variable)

## 🛠️ Technology Stack

### 🔹 Backend
- 🧠 **Machine Learning:** PyTorch (Feedforward Neural Network, Backpropagation, Dropout, L2 Regularization, Early Stopping)
- 📊 **Dataset Handling:** Pandas, NumPy
- 📈 **Model Training & Evaluation:** Scikit-learn, Matplotlib

### 🔹 Frontend
- 🌐 **UI Framework:** HTML, CSS, JavaScript

## 🚀 Installation
1️⃣ Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ro-desalination-cost-prediction.git
   cd ro-desalination-cost-prediction
   ```
2️⃣ Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the model:
   ```bash
   python train.py
   ```
4️⃣ Start the UI (if available):
   ```bash
   npm install
   npm start
   ```

## 🎯 Usage
- 🏗️ **Input plant details** via the UI.
- 📊 **Get the predicted capital cost**.
- 🌿 **Use additional features** like the carbon footprint calculator and water quality estimator.

## 🔮 Future Enhancements
- 🚀 **Improve Model Accuracy:** Fine-tune hyperparameters.
- 🌍 **Deploy Full Web Application:** Host the UI and backend.
- 🎨 **Enhance UI/UX:** Interactive graphs and visualizations.
- 💧 **Expand Water Quality Estimation:** Integrate more water quality parameters.
- 🌱 **Enhance Carbon Footprint Calculation:** Refine the model with more environmental data.

## 🤝 Contributing
Feel free to **fork** this repository, make feature enhancements, and submit a **pull request**! 🚀


## 📬 Contact
For any inquiries or collaboration, reach out at **sumanth.1si22cs053@gmail.com**.

