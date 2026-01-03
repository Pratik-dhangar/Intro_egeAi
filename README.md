# ✈️ Edge AI: Self-Healing Engine Controller 

<div align="center">  <a href="https://introegeai-9vdt2me7swysudy44ctw5i.streamlit.app/" target="_blank">

    <img src="https://img.shields.io/badge/🚀Live-Demo-success?style=for-the-badge" alt="Live Demo">

  </a> </div>



This project demonstrates an **Edge AI Digital Twin** that simulates a jet engine's sensor data and uses a TensorFlow Lite model to predict the Remaining Useful Life (RUL). If the AI predicts a failure, the system automatically intervenes to throttle the engine or perform an emergency shutdown.



## 🌟 Features



-   **Real-time Simulation**: Generates synthetic sensor data with drift and noise to mimic engine degradation.

-   **Edge AI Inference**: Uses a lightweight `tflite` model to predict RUL locally.

-   **Self-Healing Logic**:

    -   **Normal Operation**: Engine runs at 100% RPM.

    -   **Warning State**: If RUL < 40 cycles, the system throttles the engine to 75% RPM.

    -   **Critical State**: If RUL < 10 cycles, the system triggers an emergency stop (0% RPM).

-   **Interactive Dashboard**: Built with Streamlit to visualize telemetry, RUL predictions, and system status.



## 🛠️ Installation



1.  **Clone the repository**:

    ```bash

    git clone <your-repo-url>

    cd edgeAi

    ```



2.  **Install dependencies**:

    ```bash

    pip install -r requirements.txt

    ```



3.  **Ensure the model is present**:

    Make sure `engine_model.tflite` is in the root directory of the project.



## 🚀 Usage



Run the Streamlit application:



```bash

streamlit run app.py

```



The dashboard will open in your default web browser.



1.  Use the **Control Panel** on the left to adjust the simulation speed.

2.  Click **Start Engine Simulation** to begin the monitoring process.

3.  Watch the **Live Telemetry** and **Chart** to see how the AI responds to sensor degradation.



## 📂 Project Structure



-   `app.py`: Main application script containing the simulation, UI, and control logic.

-   `engine_model.tflite`: The pre-trained TensorFlow Lite model used for inference.

-   `requirements.txt`: List of Python dependencies.

-   `README.md`: Project documentation.



## 🧠 How it Works



1.  **Data Generation**: The app simulates 14 sensors. Over time, values drift from their base levels to simulate wear and tear.

2.  **Preprocessing**: Sensor data is normalized and formatted for the model.

3.  **Inference**: The TFLite interpreter runs the model on the current sensor snapshot.

4.  **Decision Making**: The predicted RUL is compared against safety thresholds to determine the engine's operating state.

## 🤖 Model Training Logic

The `engine_model.tflite` was created using a synthetic dataset that mimics jet engine sensor behavior. Here's the core training approach:

### 1. Synthetic Data Generation
```python
# Simulate 100 engines with varying lifespans (120-350 cycles)
for each engine:
    max_life = random(120, 350)
    for time in range(1, max_life):
        degradation = time / max_life
        RUL = max_life - time
        
        # Generate 14 sensor readings
        for each sensor:
            base_value = 1000 + (sensor_index * 100)
            drift = ±50 * degradation  # Even sensors drift up, odd drift down
            noise = random_normal(0, 2)
            sensor_value = base_value + drift + noise
```

**Key Insight**: As engines age (degradation → 1), sensor values drift from baseline, creating predictable patterns.

### 2. Neural Network Architecture
```python
model = Sequential([
    Dense(32, activation='relu', input_shape=(14,)),  # 14 sensor inputs
    Dense(16, activation='relu'),                      # Hidden layer
    Dense(1)                                           # RUL prediction output
])
```

### 3. Training Process
- **Input**: 14 normalized sensor readings (s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21)
- **Output**: Remaining Useful Life (RUL) in cycles
- **Normalization**: `(X - X_min) / (X_max - X_min)` scales all values between 0 and 1
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

### 4. Conversion to Edge AI Format
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Result: Lightweight model optimized for edge devices
```

The TFLite format reduces model size while maintaining prediction accuracy, making it ideal for deployment on resource-constrained edge devices.
