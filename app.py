import streamlit as st
import numpy as np
import time

# Use only TFLite interpreter instead of full TensorFlow
try:
    from tensorflow import lite as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Edge AI Digital Twin", layout="wide")

# --- 2. CONFIGURATION ---
# We calculate the Base, Min, and Max values to match the logic 
# used in the "Foolproof" training script.
NUM_SENSORS = 14
BASE_VALS = np.array([1000 + (i * 100) for i in range(NUM_SENSORS)])
# We add a buffer of +/- 60 to cover noise, just like the training scaling
MIN_VALS = BASE_VALS - 60
MAX_VALS = BASE_VALS + 60

# --- 3. LOAD THE BRAIN ---
@st.cache_resource
def load_model():
    # Make sure 'engine_model.tflite' is in the same folder!
    try:
        interpreter = tflite.Interpreter(model_path="engine_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'engine_model.tflite' is in this folder.")
        return None

interpreter = load_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- 4. THE SIMULATION UI ---
    st.title("✈️ Edge AI: Self-Healing Engine Controller")
    st.markdown("This dashboard simulates an Edge AI chip monitoring a Jet Engine. If the AI predicts failure, it **automatically** throttles the engine to save it.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Control Panel")
        st.markdown("""
        **What is this?**  
        The control panel allows you to configure and start the engine simulation.
        
        **Role:**  
        - Adjust the simulation speed to control how fast the engine degrades
        - Initiate the real-time monitoring and AI-driven control system
        - Watch as the Edge AI makes autonomous decisions to protect the engine
        """)
        simulation_speed = st.slider("Simulation Speed (Lower = Faster, Higher = Slower)", 0.1, 2.0, 0.5)
        st.caption("⬅️ Fast (0.1s delay) | Slow (2.0s delay) ➡️")
        start_btn = st.button("Start Engine Simulation", type="primary")

    with col2:
        st.header("Live Telemetry")
        st.markdown("""
        **What is this?**  
        Real-time metrics from the engine monitoring system and AI predictions.
        
        **Role:**  
        - **Predicted Life (RUL)**: Remaining Useful Life estimated by the AI model
        - **Engine RPM**: Current throttle level (AI automatically reduces RPM when failure is predicted)
        - **Sensor 1**: Live temperature reading from the first engine sensor
        - **Status Alerts**: AI-driven warnings and automatic safety interventions
        """)
        metric_placeholder = st.empty()
        alert_placeholder = st.empty()
    
    # Graph section below control panel
    st.markdown("---")  # Visual separator
    st.header("📊 Degradation Trend")
    st.markdown("**Real-time graph showing the predicted Remaining Useful Life (RUL) over engine cycles.**")
    chart_placeholder = st.empty()

    # --- 5. SIMULATOR LOGIC ---
    if start_btn:
        rul_history = []
        
        # Pre-calculate drift directions to match training (Even indices go up, Odd go down)
        drift_directions = np.array([(i % 2 * 2 - 1) for i in range(NUM_SENSORS)])
        
        for cycle in range(100):
            # A. GENERATE FAKE SENSOR DATA
            degradation_factor = cycle / 100.0 
            
            # Calculate current sensor values
            current_sensors = BASE_VALS + (drift_directions * 50 * degradation_factor)
            
            # Add random noise
            noise = np.random.normal(0, 2, NUM_SENSORS)
            current_sensors += noise
            
            # B. PREPROCESS (NORMALIZE)
            input_data = (current_sensors - MIN_VALS) / (MAX_VALS - MIN_VALS)
            input_data = np.clip(input_data, 0, 1) 
            input_data = input_data.astype(np.float32).reshape(1, NUM_SENSORS)

            # C. AI INFERENCE
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predicted_rul = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # D. CONTROL LOGIC
            engine_status = "NORMAL"
            engine_rpm = 100 
            
            if predicted_rul < 40:
                 engine_status = "WARNING: THROTTLING"
                 engine_rpm = 75
            
            if predicted_rul < 10:
                 engine_status = "CRITICAL: EMERGENCY STOP"
                 engine_rpm = 0

            # E. UPDATE UI
            with metric_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Life (RUL)", f"{predicted_rul:.1f} Cycles")
                c2.metric("Engine RPM (Action)", f"{engine_rpm}%")
                c1_val = current_sensors[0]
                c3.metric("Sensor 1 (Temp)", f"{c1_val:.1f}") 

            with alert_placeholder:
                if engine_rpm == 75:
                    st.warning(f"⚠️ DETECTED DEGRADATION (RUL {predicted_rul:.0f}). AUTO-THROTTLING ACTIVE.")
                elif engine_rpm == 0:
                    st.error(f"🛑 FAILURE IMMINENT. EMERGENCY SHUTDOWN.")
                else:
                    st.success("✅ ENGINE HEALTHY")

            # Update Chart with axis labels
            rul_history.append(predicted_rul)
            import pandas as pd
            chart_data = pd.DataFrame(rul_history, columns=["Remaining Useful Life (Cycles)"])
            chart_placeholder.line_chart(chart_data, x_label="Engine Cycles", y_label="RUL (Cycles)")

            time.sleep(simulation_speed)
else:
    st.warning("Please ensure 'engine_model.tflite' is in the same directory and restart.")