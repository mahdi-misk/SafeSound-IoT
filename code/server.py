import os
import wave
import traceback
import pickle
import warnings
import numpy as np
import librosa
import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
import uvicorn
import shutil
import tempfile

warnings.filterwarnings('ignore')

# Set ffmpeg path explicitly for pydub
AudioSegment.converter = r"C:\Users\mahdi\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"

app = FastAPI()

# ==================== AI Model Initialization ====================
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ai_training', 'audio_classification_model.onnx')
PKL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'AI', 'project-main', 'models', 'q_table_rf_model.pkl')
H5_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ai_training', 'audio_classification_model.h5')

ai_model_type = None  # Can be 'onnx', 'h5', or 'pkl'
onnx_session = None
ai_model = None
ai_scaler = None

try:
    if os.path.exists(ONNX_MODEL_PATH):
        print(f"Loading Keras CNN ONNX Model from: {ONNX_MODEL_PATH}")
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
        ai_model_type = 'onnx'
        print("Keras CNN ONNX Model loaded successfully!")
    
    # Try H5 only if ONNX is not loaded
    if ai_model_type is None and os.path.exists(H5_MODEL_PATH):
        print(f"Loading Keras H5 Model from: {H5_MODEL_PATH}")
        try:
            import tensorflow as tf
            ai_model = tf.keras.models.load_model(H5_MODEL_PATH)
            ai_model_type = 'h5'
            print("Keras H5 Model loaded successfully!")
        except ImportError:
            print("⚠️ TensorFlow is not installed. Cannot load H5 model.")

    # Fallback to PKL if neither ONNX nor H5 loaded successfully
    if ai_model_type is None and os.path.exists(PKL_MODEL_PATH):
        print(f"ONNX/H5 Model not found/loaded. Loading fallback RF Model from: {PKL_MODEL_PATH}")
        with open(PKL_MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            ai_model = model_data['model']
            ai_scaler = model_data['scaler']
        ai_model_type = 'pkl'
        print("Fallback RF Model loaded successfully!")
    else:
        print("No AI Model found in workspace.")
except Exception as e:
    print(f"Error loading AI Model: {e}")
    traceback.print_exc()

def extract_features(audio_path):
    """Extract 16 audio features from the file for RF model"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3)
        
        features = []
        
        # 1-13: MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(mfccs.mean(axis=1))
        
        # 14: Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(spectral_centroid.mean())
        
        # 15: Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(zcr.mean())
        
        # 16: RMS Energy
        rms = librosa.feature.rms(y=y)
        features.append(rms.mean())
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def extract_mel_spectrogram(file_path):
    """Extract Mel-Spectrogram features for the CNN ONNX model"""
    try:
        # Load audio at 16kHz for max 2.0s duration
        y, sr = librosa.load(file_path, sr=16000, duration=2.0)
        
        # Extract Mel-spectrogram and convert to decibels
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize image size to 128x128
        IMG_WIDTH = 128
        if mel_spec_db.shape[1] < IMG_WIDTH:
            pad_width = IMG_WIDTH - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0,0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :IMG_WIDTH]
            
        return mel_spec_db
    except Exception as e:
        print(f"Error extracting Mel-spectrogram: {e}")
        return None

def extract_mfcc_40(file_path):
    """Extract 40 MFCC features for 1D CNN"""
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting MFCC 40: {e}")
        return None
# =================================================================

# Create templates directory if it doesn't exist
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global state
connected_esp32: WebSocket = None
connected_device_name = "ESP32"
is_recording = False
audio_buffer = bytearray()
sample_rate = 8000 # Matches ESP32 sampling rate
ai_status_message = ""

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "is_recording": is_recording})

def run_inference_sync(chunk: bytearray, sr: int):
    import time
    global connected_device_name
    
    # Save audio permanently to PC
    os.makedirs("saved_audio", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_safe = "".join([c if c.isalnum() else "_" for c in connected_device_name])
    saved_wav_path = os.path.join("saved_audio", f"{device_safe}_audio_{timestamp}.wav")
    
    result = {"status": "error", "prediction": None, "device": "Unknown"}
    try:
        with wave.open(saved_wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(sr)
            wf.writeframes(chunk)
            
        print(f"✅ Audio chunk saved to PC at: {saved_wav_path}")
            
        if ai_model_type == 'onnx' and onnx_session is not None:
            input_name = onnx_session.get_inputs()[0].name
            input_shape = onnx_session.get_inputs()[0].shape
            
            raw_output = None
            if len(input_shape) >= 3 and input_shape[1] == 40:
                features = extract_mfcc_40(saved_wav_path)
                if features is not None:
                    input_data = features.reshape(1, 40, 1).astype(np.float32)
                    raw_output = onnx_session.run(None, {input_name: input_data})[0]
            else:
                mel_features = extract_mel_spectrogram(saved_wav_path)
                if mel_features is not None:
                    input_data = mel_features.reshape(1, 128, 128, 1).astype(np.float32)
                    raw_output = onnx_session.run(None, {input_name: input_data})[0]
            
            if raw_output is not None:
                num_classes = raw_output.shape[1]
                if num_classes == 8:
                    raw_pred = raw_output[0]
                    prediction_class = int(np.argmax(raw_pred))
                    is_abnormal = 1 if (prediction_class % 2 != 0) else 0
                    devices_list = ["Fan", "Pump", "Slider", "Valve"]
                    detected_device = devices_list[prediction_class // 2]
                    
                    result["status"] = "success"
                    result["prediction"] = is_abnormal
                    result["device"] = detected_device
                else:
                    raw_pred = raw_output[0][0]
                    prediction = 1 if raw_pred >= 0.5 else 0
                    result["status"] = "success"
                    result["prediction"] = prediction
                    result["device"] = "Unknown"
        elif ai_model_type == 'h5' and ai_model is not None:
            # Run Keras H5 Inference
            import tensorflow as tf
            mel_features = extract_mel_spectrogram(saved_wav_path)
            if mel_features is not None:
                # Shape: (1, 128, 128, 1)
                input_data = mel_features.reshape(1, 128, 128, 1).astype(np.float32)
                raw_output = ai_model.predict(input_data, verbose=0)
                
                num_classes = raw_output.shape[1]
                if num_classes == 8:
                    raw_pred = raw_output[0]
                    prediction_class = int(np.argmax(raw_pred))
                    is_abnormal = 1 if (prediction_class % 2 != 0) else 0
                    devices_list = ["Fan", "Pump", "Slider", "Valve"]
                    detected_device = devices_list[prediction_class // 2]
                    
                    result["status"] = "success"
                    result["prediction"] = is_abnormal
                    result["device"] = detected_device
                else:
                    raw_pred = raw_output[0][0]
                    prediction = 1 if raw_pred >= 0.5 else 0
                    result["status"] = "success"
                    result["prediction"] = prediction
                    result["device"] = "Unknown"
        elif ai_model_type == 'pkl' and ai_model is not None and ai_scaler is not None:
            # Run Fallback RF Inference
            features = extract_features(saved_wav_path)
            if features is not None:
                features_scaled = ai_scaler.transform([features])
                prediction = ai_model.predict(features_scaled)[0]
                result["status"] = "success"
                result["prediction"] = int(prediction)
                result["device"] = "Unknown"
        else:
            print("No active model loaded for inference.")
    except Exception as e:
        print(f"AI inference error: {e}")
    # We no longer delete the saved_wav_path here, so it is kept permanently.
    return result

async def process_audio_chunk(chunk: bytearray, websocket: WebSocket, sr: int):
    global ai_status_message
    try:
        await websocket.send_text("STATE_PROCESSING")
        ai_status_message = "Analyzing..."
        
        result = await asyncio.to_thread(run_inference_sync, chunk, sr)
        
        if result["status"] == "success":
            prediction = result["prediction"]
            device = result.get("device", "Unknown")
            device_prefix = f"[{device}] " if device != "Unknown" else ""
            
            if prediction == 1:
                ai_status_message = f"<b>🤖 AI:</b> {device_prefix}<span style='color:red'>⚠️ Anomaly Detected!</span>"
                await websocket.send_text("STATE_ABNORMAL")
                print(f"⚠️ {device_prefix}Anomaly Detected!")
            else:
                ai_status_message = f"<b>🤖 AI:</b> {device_prefix}<span style='color:green'>✅ Sound is normal.</span>"
                await websocket.send_text("STATE_NORMAL")
                print(f"✅ {device_prefix}Sound is normal.")
    except Exception as e:
        print(f"Error in process_audio_chunk: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global connected_esp32, connected_device_name, is_recording, audio_buffer, ai_status_message
    device_param = websocket.query_params.get("device", "ESP32")
    connected_device_name = device_param
    await websocket.accept()
    connected_esp32 = websocket
    print(f"Device Connected: {connected_device_name}!")
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                # Received audio chunk
                if is_recording:
                    audio_buffer.extend(data["bytes"])
                    # Process every 3 seconds (24000 bytes at 8000Hz)
                    if len(audio_buffer) >= 24000:
                        chunk = audio_buffer[:24000]
                        # Preserve leftover bytes to avoid audio data loss
                        audio_buffer = bytearray(audio_buffer[24000:])
                        
                        # Process in background task to avoid blocking the websocket loop
                        asyncio.create_task(process_audio_chunk(chunk, websocket, sample_rate))
 
            elif "text" in data:
                print(f"Message from {connected_device_name}: {data['text']}")
    except WebSocketDisconnect:
        print(f"Device Disconnected: {connected_device_name}")
        connected_esp32 = None

@app.post("/api/pump/{action}")
async def control_pump(action: str):
    global connected_esp32
    if connected_esp32 is None:
        return {"status": "error", "message": "Device is not connected"}
    
    if action in ["on", "off"]:
        command = f"PUMP_{action.upper()}"
        try:
            await connected_esp32.send_text(command)
            return {"status": "success", "message": f"Pump turned {'on' if action == 'on' else 'off'}"}
        except Exception as e:
            return {"status": "error", "message": f"Transmission error: {e}"}
    return {"status": "error", "message": "Invalid action"}

@app.get("/api/info")
async def get_info():
    global ai_model_type, connected_esp32, connected_device_name
    return {
        "status": "success",
        "model_type": ai_model_type if ai_model_type else "None",
        "esp32_connected": connected_esp32 is not None,
        "device_name": connected_device_name if connected_esp32 is not None else None
    }

@app.get("/api/status")
async def get_status():
    global ai_status_message, is_recording
    if is_recording:
        return {"status": "success", "message": ai_status_message}
    return {"status": "success", "message": ""}

@app.post("/api/record/{action}")
async def control_recording(action: str):
    global is_recording, audio_buffer, connected_esp32, ai_status_message
    if action == "start":
        is_recording = True
        audio_buffer = bytearray()
        ai_status_message = "Recording and awaiting analysis..."
        if connected_esp32:
            try:
                await connected_esp32.send_text("START_RECORDING")
            except Exception as e:
                print(f"Error sending START command: {e}")
        return {"status": "success", "message": "Continuous recording started"}
    elif action == "stop":
        is_recording = False
        ai_status_message = "Recording stopped."
        audio_buffer = bytearray()
        if connected_esp32:
            try:
                await connected_esp32.send_text("STOP_RECORDING")
            except Exception as e:
                print(f"Error sending STOP command: {e}")
        return {"status": "success", "message": "Continuous recording stopped"}
    
    return {"status": "error", "message": "Invalid action"}

@app.post("/api/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    global ai_model_type, onnx_session, ai_model, ai_scaler
    
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"upload_{datetime.now().timestamp()}_{file.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    result = {"status": "error", "prediction": None, "device": "Unknown", "message": "No prediction made"}
    wav_path = temp_file_path + ".wav"
    
    try:
        audio = AudioSegment.from_file(temp_file_path)
        audio.export(wav_path, format="wav")
        
        if ai_model_type == 'onnx' and onnx_session is not None:
            input_name = onnx_session.get_inputs()[0].name
            input_shape = onnx_session.get_inputs()[0].shape
            
            raw_output = None
            if len(input_shape) >= 3 and input_shape[1] == 40:
                features = extract_mfcc_40(wav_path)
                if features is not None:
                    input_data = features.reshape(1, 40, 1).astype(np.float32)
                    raw_output = onnx_session.run(None, {input_name: input_data})[0]
            else:
                mel_features = extract_mel_spectrogram(wav_path)
                if mel_features is not None:
                    input_data = mel_features.reshape(1, 128, 128, 1).astype(np.float32)
                    raw_output = onnx_session.run(None, {input_name: input_data})[0]
            
            if raw_output is not None:
                num_classes = raw_output.shape[1]
                if num_classes == 8:
                    raw_pred = raw_output[0]
                    prediction_class = int(np.argmax(raw_pred))
                    is_abnormal = 1 if (prediction_class % 2 != 0) else 0
                    devices_list = ["Fan", "Pump", "Slider", "Valve"]
                    detected_device = devices_list[prediction_class // 2]
                    
                    result["status"] = "success"
                    result["prediction"] = is_abnormal
                    result["device"] = detected_device
                    result["message"] = f"Detected {detected_device} - Anomaly!" if is_abnormal == 1 else f"Detected {detected_device} - Normal."
                else:
                    raw_pred = raw_output[0][0]
                    prediction = 1 if raw_pred >= 0.5 else 0
                    result["status"] = "success"
                    result["prediction"] = prediction
                    result["device"] = "Unknown"
                    result["message"] = "Anomaly Detected!" if prediction == 1 else "Sound is normal."
        elif ai_model_type == 'h5' and ai_model is not None:
            import tensorflow as tf
            mel_features = extract_mel_spectrogram(wav_path)
            if mel_features is not None:
                input_data = mel_features.reshape(1, 128, 128, 1).astype(np.float32)
                raw_output = ai_model.predict(input_data, verbose=0)
                
                num_classes = raw_output.shape[1]
                if num_classes == 8:
                    raw_pred = raw_output[0]
                    prediction_class = int(np.argmax(raw_pred))
                    is_abnormal = 1 if (prediction_class % 2 != 0) else 0
                    devices_list = ["Fan", "Pump", "Slider", "Valve"]
                    detected_device = devices_list[prediction_class // 2]
                    
                    result["status"] = "success"
                    result["prediction"] = is_abnormal
                    result["device"] = detected_device
                    result["message"] = f"Detected {detected_device} - Anomaly!" if is_abnormal == 1 else f"Detected {detected_device} - Normal."
                else:
                    raw_pred = raw_output[0][0]
                    prediction = 1 if raw_pred >= 0.5 else 0
                    result["status"] = "success"
                    result["prediction"] = prediction
                    result["device"] = "Unknown"
                    result["message"] = "Anomaly Detected!" if prediction == 1 else "Sound is normal."
        elif ai_model_type == 'pkl' and ai_model is not None and ai_scaler is not None:
            features = extract_features(wav_path)
            if features is not None:
                features_scaled = ai_scaler.transform([features])
                prediction = ai_model.predict(features_scaled)[0]
                result["status"] = "success"
                result["prediction"] = int(prediction)
                result["device"] = "Unknown"
                result["message"] = "Anomaly Detected!" if prediction == 1 else "Sound is normal."
        else:
            result["message"] = "No active AI model loaded on server."
            
    except Exception as e:
        result["message"] = f"Inference failed: {str(e)}"
        print(f"Error predicting uploaded audio: {e}")
        traceback.print_exc()
    finally:
        for p in [temp_file_path, wav_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                print(f"Error deleting temp file {p}: {e}")
                
    return result

if __name__ == "__main__":
    import socket
    from zeroconf import ServiceInfo, Zeroconf
    
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
            
    local_ip = get_local_ip()
    
    # Setup mDNS
    zeroconf = Zeroconf()
    info = ServiceInfo(
        "_ws._tcp.local.",
        "AI WebSocket Server._ws._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=8000,
        server="ai-server.local.",
    )
    zeroconf.register_service(info)
    
    print("\n" + "="*50)
    print(f" [Web] Web Interface running at: http://{local_ip}:8000")
    print(f" [ESP] ESP32 should connect to: ws://ai-server.local:8000/ws (mDNS)")
    print(f"       Or use IP directly: ws://{local_ip}:8000/ws")
    print("="*50 + "\n")
    
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    finally:
        zeroconf.unregister_service(info)
        zeroconf.close()
