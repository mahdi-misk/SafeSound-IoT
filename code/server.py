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

def clear_saved_audio():
    paths_to_clear = [
        os.path.join(os.path.dirname(__file__), 'saved_audio'),
        os.path.join(os.path.dirname(__file__), '..', 'saved_audio')
    ]
    for path in paths_to_clear:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                for filename in os.listdir(abs_path):
                    file_path = os.path.join(abs_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"Cleared old audio files from: {abs_path}")
            except Exception as e:
                print(f"Failed to clear {abs_path}. Reason: {e}")

# clear_saved_audio()  # Disabled: keep old audio files for debugging

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
            print("TensorFlow is not installed. Cannot load H5 model.")

    # Fallback to PKL if neither ONNX nor H5 loaded successfully
    if ai_model_type is None and os.path.exists(PKL_MODEL_PATH):
        print(f"ONNX/H5 Model not found/loaded. Loading fallback RF Model from: {PKL_MODEL_PATH}")
        with open(PKL_MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            ai_model = model_data['model']
            ai_scaler = model_data['scaler']
        ai_model_type = 'pkl'
        print("Fallback RF Model loaded successfully!")
    elif ai_model_type is None:
        print("No AI Model found in workspace.")
except Exception as e:
    print(f"Error loading AI Model: {e}")
    traceback.print_exc()

def extract_features(y, sr=16000):
    """Extract 16 audio features from array for RF model"""
    try:
        max_samples = int(5.0 * sr)
        if len(y) > max_samples:
            y = y[:max_samples]
            
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

def extract_mel_spectrogram(y, sr=16000):
    """Extract Mel-Spectrogram features from array"""
    try:
        max_samples = int(2.0 * sr)
        if len(y) > max_samples:
            y = y[:max_samples]
            
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

def extract_mfcc_40(y, sr=16000):
    """Extract 40 MFCC features from array"""
    try:
        max_samples = int(3.0 * sr)
        if len(y) > max_samples:
            y = y[:max_samples]
            
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting MFCC 40: {e}")
        return None
# =================================================================


import collections
import time

# Create templates directory if it doesn't exist
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ==================== Live Monitoring Settings ====================
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
WINDOW_SECONDS = 10
HOP_SECONDS = 1

WINDOW_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * WINDOW_SECONDS
HOP_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * HOP_SECONDS
SAVE_DEBUG_WAV = False

# Global state
connected_esp32 = None
connected_device_name = "device_1"
is_recording = False
rolling_buffer = bytearray()
bytes_since_last_inference = 0
inference_running = False
ai_status_message = "Waiting to start..."

# Status tracking
recent_predictions = collections.deque(maxlen=5)
recent_devices = collections.deque(maxlen=7)
current_smoothed_state = 0
current_raw_state = 0
current_confidence = 0.0
last_inference_latency_ms = 0
total_received_seconds = 0.0

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "is_recording": is_recording})

def run_inference_sync(chunk: bytearray, sr: int):
    global connected_device_name
    
    start_time = time.time()
    
    if SAVE_DEBUG_WAV:
        os.makedirs("saved_audio", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_safe = "".join([c if c.isalnum() else "_" for c in connected_device_name])
        saved_wav_path = os.path.join("saved_audio", f"{device_safe}_audio_{timestamp}.wav")
        try:
            with wave.open(saved_wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(chunk)
            print(f"Audio chunk saved to PC at: {saved_wav_path}")
        except Exception as e:
            print(f"Error saving wav: {e}")
            
    result = {"status": "error", "prediction": None, "device": "Unknown", "confidence": 0.0, "latency_ms": 0}
    try:
        if ai_model_type == 'onnx' and onnx_session is not None:
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            y_16k = librosa.resample(y=audio_float, orig_sr=sr, target_sr=16000)
            target_sr = 16000

            input_name = onnx_session.get_inputs()[0].name
            input_shape = onnx_session.get_inputs()[0].shape
            
            raw_output = None
            if len(input_shape) >= 3 and input_shape[1] == 40:
                features = extract_mfcc_40(y_16k, target_sr)
                if features is not None:
                    input_data = features.reshape(1, 40, 1).astype(np.float32)
                    raw_output = onnx_session.run(None, {input_name: input_data})[0]
            else:
                mel_features = extract_mel_spectrogram(y_16k, target_sr)
                if mel_features is not None:
                    input_data = mel_features.reshape(1, 128, 128, 1).astype(np.float32)
                    raw_output = onnx_session.run(None, {input_name: input_data})[0]
            
            if raw_output is not None:
                num_classes = raw_output.shape[1]
                if num_classes == 8:
                    raw_pred = raw_output[0]
                    prediction_class = int(np.argmax(raw_pred))
                    confidence = float(np.max(raw_pred))
                    is_abnormal = 1 if (prediction_class % 2 != 0) else 0
                    devices_list = ["Fan", "Pump", "Slider", "Valve"]
                    detected_device = devices_list[prediction_class // 2]
                    
                    result["status"] = "success"
                    result["prediction"] = is_abnormal
                    result["device"] = detected_device
                    result["confidence"] = confidence
                else:
                    raw_pred = raw_output[0][0]
                    prediction = 1 if raw_pred >= 0.5 else 0
                    confidence = float(raw_pred if prediction == 1 else 1.0 - raw_pred)
                    result["status"] = "success"
                    result["prediction"] = prediction
                    result["device"] = "Unknown"
                    result["confidence"] = confidence
        elif ai_model_type == 'h5' and ai_model is not None:
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            y_16k = librosa.resample(y=audio_float, orig_sr=sr, target_sr=16000)
            target_sr = 16000
            
            mel_features = extract_mel_spectrogram(y_16k, target_sr)
            if mel_features is not None:
                input_data = mel_features.reshape(1, 128, 128, 1).astype(np.float32)
                raw_output = ai_model.predict(input_data, verbose=0)
                
                num_classes = raw_output.shape[1]
                if num_classes == 8:
                    raw_pred = raw_output[0]
                    prediction_class = int(np.argmax(raw_pred))
                    confidence = float(np.max(raw_pred))
                    is_abnormal = 1 if (prediction_class % 2 != 0) else 0
                    devices_list = ["Fan", "Pump", "Slider", "Valve"]
                    detected_device = devices_list[prediction_class // 2]
                    
                    result["status"] = "success"
                    result["prediction"] = is_abnormal
                    result["device"] = detected_device
                    result["confidence"] = confidence
                else:
                    raw_pred = raw_output[0][0]
                    prediction = 1 if raw_pred >= 0.5 else 0
                    confidence = float(raw_pred if prediction == 1 else 1.0 - raw_pred)
                    result["status"] = "success"
                    result["prediction"] = prediction
                    result["device"] = "Unknown"
                    result["confidence"] = confidence
        elif ai_model_type == 'pkl' and ai_model is not None and ai_scaler is not None:
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            y_16k = librosa.resample(y=audio_float, orig_sr=sr, target_sr=16000)
            target_sr = 16000
            
            features = extract_features(y_16k, target_sr)
            if features is not None:
                features_scaled = ai_scaler.transform([features])
                probs = ai_model.predict_proba(features_scaled)[0]
                prediction = int(np.argmax(probs))
                confidence = float(np.max(probs))
                result["status"] = "success"
                result["prediction"] = prediction
                result["device"] = "Unknown"
                result["confidence"] = confidence
        else:
            pass
    except Exception as e:
        print(f"AI inference error: {e}")
        
    end_time = time.time()
    result["latency_ms"] = int((end_time - start_time) * 1000)
    return result

async def process_audio_chunk_async(chunk: bytearray, sr: int):
    global ai_status_message, is_recording, inference_running
    global recent_predictions, recent_devices, current_smoothed_state, current_raw_state
    global current_confidence, last_inference_latency_ms
    
    try:
        result = await asyncio.to_thread(run_inference_sync, chunk, sr)
        
        if result["status"] == "success":
            prediction = result["prediction"]
            device = result.get("device", "Unknown")
            confidence = result.get("confidence", 0.0)
            latency = result.get("latency_ms", 0)
            
            current_raw_state = prediction
            current_confidence = confidence
            last_inference_latency_ms = latency
            
            recent_predictions.append(prediction)
            if device != "Unknown":
                recent_devices.append(device)
            
            smoothed_prediction = 1 if sum(recent_predictions) >= 3 else 0
            current_smoothed_state = smoothed_prediction
            
            smoothed_device = device
            if recent_devices:
                smoothed_device = collections.Counter(recent_devices).most_common(1)[0][0]

            device_prefix = f"[{smoothed_device}] " if smoothed_device != "Unknown" else ""
            
            raw_str = "Abnormal" if current_raw_state == 1 else "Normal"
            smooth_str = "Abnormal" if current_smoothed_state == 1 else "Normal"
            print(f"[INFERENCE] Window: {len(chunk)}B | Raw: {raw_str} ({confidence:.2f}) | Smoothed: {smooth_str} | Latency: {latency}ms | Total Rx: {total_received_seconds:.1f}s")
            
            if smoothed_prediction == 1:
                ai_status_message = f"<b>🤖 AI:</b> {device_prefix}<span style='color:red'>⚠️ Anomaly Detected!</span>"
                if connected_esp32:
                    try:
                        await connected_esp32.send_text("STATE_ABNORMAL")
                    except Exception as e:
                        print(f"Error sending STATE_ABNORMAL: {e}")
            else:
                ai_status_message = f"<b>🤖 AI:</b> {device_prefix}<span style='color:green'>✅ Sound is normal.</span>"
                if connected_esp32:
                    try:
                        await connected_esp32.send_text("STATE_NORMAL")
                    except Exception as e:
                        print(f"Error sending STATE_NORMAL: {e}")
    except Exception as e:
        print(f"Error in process_audio_chunk: {e}")
    finally:
        inference_running = False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global connected_esp32, connected_device_name, is_recording
    global rolling_buffer, bytes_since_last_inference, inference_running
    global total_received_seconds, ai_status_message
    
    device_param = websocket.query_params.get("device", "ESP32")
    connected_device_name = device_param
    await websocket.accept()
    connected_esp32 = websocket
    print(f"[WS] Device Connected: {connected_device_name}!")
    
    bytes_received_total = 0
    chunks_count = 0
    
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                incoming_len = len(data["bytes"])
                bytes_received_total += incoming_len
                
                if is_recording:
                    total_received_seconds += incoming_len / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                    rolling_buffer.extend(data["bytes"])
                    bytes_since_last_inference += incoming_len
                    
                    if len(rolling_buffer) > WINDOW_BYTES:
                        rolling_buffer = rolling_buffer[-WINDOW_BYTES:]
                    
                    if len(rolling_buffer) < WINDOW_BYTES:
                        ai_status_message = "Collecting 10s audio window..."
                        chunks_count += 1
                        if chunks_count % 50 == 0:
                            print(f"[WS] Buffering: {len(rolling_buffer)}/{WINDOW_BYTES} bytes...")
                    else:
                        if bytes_since_last_inference >= HOP_BYTES:
                            if inference_running:
                                bytes_since_last_inference = 0
                            else:
                                chunk_to_process = bytearray(rolling_buffer)
                                bytes_since_last_inference = 0
                                inference_running = True
                                asyncio.create_task(process_audio_chunk_async(chunk_to_process, SAMPLE_RATE))
                else:
                    if chunks_count % 100 == 0:
                        print(f"[WS] Receiving data but NOT recording. Got {incoming_len} bytes")
                    chunks_count += 1
            elif "text" in data:
                print(f"[WS] Text from {connected_device_name}: {data['text']}")
    except WebSocketDisconnect:
        print(f"[WS] Device Disconnected: {connected_device_name} (received {bytes_received_total} total bytes)")
        connected_esp32 = None

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
    global rolling_buffer, current_raw_state, current_smoothed_state
    global current_confidence, last_inference_latency_ms, total_received_seconds
    
    buffer_seconds = len(rolling_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    
    return {
        "status": "success", 
        "message": ai_status_message if is_recording else "",
        "is_recording": is_recording,
        "buffer_seconds": buffer_seconds,
        "raw_state": current_raw_state,
        "smoothed_state": current_smoothed_state,
        "confidence": current_confidence,
        "latency_ms": last_inference_latency_ms,
        "total_received_seconds": total_received_seconds
    }

@app.post("/api/record/{action}")
async def control_recording(action: str):
    global is_recording, rolling_buffer, connected_esp32, ai_status_message
    global bytes_since_last_inference, inference_running, recent_predictions, recent_devices
    global total_received_seconds
    
    if action == "start":
        is_recording = True
        rolling_buffer = bytearray()
        bytes_since_last_inference = 0
        inference_running = False
        recent_predictions.clear()
        recent_devices.clear()
        total_received_seconds = 0.0
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
        if connected_esp32:
            try:
                await connected_esp32.send_text("STOP_RECORDING")
            except Exception as e:
                print(f"Error sending STOP command: {e}")
        return {"status": "success", "message": "Continuous recording stopped"}
    
    return {"status": "error", "message": "Invalid action"}

@app.post("/api/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load with librosa at target sample rate 16000
        y, sr = librosa.load(tmp_path, sr=16000)
        os.remove(tmp_path)

        # Convert to int16 bytes to feed into run_inference_sync
        audio_int16 = (y * 32767).astype(np.int16)
        chunk = audio_int16.tobytes()

        # Run the existing inference pipeline
        result = await asyncio.to_thread(run_inference_sync, chunk, 16000)

        if result["status"] == "success":
            pred = result["prediction"]
            device = result.get("device", "Unknown")
            conf = result.get("confidence", 0.0)
            
            state_str = "Abnormal" if pred == 1 else "Normal"
            device_prefix = f"[{device}] " if device != "Unknown" else ""
            msg = f"{device_prefix}{state_str} (Conf: {conf*100:.1f}%)"
            
            return {
                "status": "success",
                "prediction": pred,
                "message": msg
            }
        else:
            return {"status": "error", "prediction": -1, "message": "Inference failed."}

    except Exception as e:
        print(f"Error in predict-audio: {e}")
        return {"status": "error", "prediction": -1, "message": str(e)}

if __name__ == "__main__":
    import socket
    from zeroconf import ServiceInfo, Zeroconf
    
    def get_local_ip():
        # Try ESP32 AP gateway first (for AP mode), then internet
        for target in ["192.168.4.1", "8.8.8.8"]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(1)
                s.connect((target, 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                pass
        return "192.168.4.2"
            
    local_ip = get_local_ip()
    
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
