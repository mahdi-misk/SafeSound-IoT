import asyncio
import websockets
import wave
import argparse
import time
import requests
import sys
import os

CHUNK_SAMPLES = 1024
CHUNK_BYTES = 2048
SLEEP_SECONDS = 1024 / 16000.0

def verify_wav_format(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
        
    try:
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            
            print(f"Checking WAV: {channels} channels, {sample_width} bytes/sample, {framerate} Hz")
            
            if channels != 1:
                print("Error: WAV must be mono (1 channel).")
                sys.exit(1)
            if sample_width != 2:
                print("Error: WAV must be int16 (2 bytes per sample).")
                sys.exit(1)
            if framerate != 16000:
                print("Error: WAV must be 16000 Hz.")
                sys.exit(1)
                
            return wf.readframes(wf.getnframes())
    except Exception as e:
        print(f"Error reading WAV: {e}")
        sys.exit(1)

async def stream_audio(ws, audio_data, loop_audio):
    global is_recording
    while True:
        pos = 0
        while pos < len(audio_data):
            if not is_recording:
                await asyncio.sleep(0.1)
                continue
                
            chunk = audio_data[pos:pos+CHUNK_BYTES]
            if not chunk:
                break
                
            try:
                await ws.send(chunk)
                pos += CHUNK_BYTES
                await asyncio.sleep(SLEEP_SECONDS)
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket closed during streaming.")
                return
                
        if not loop_audio:
            print("Finished streaming audio file.")
            break
        else:
            print("Looping audio file...")

async def receive_messages(ws):
    global is_recording
    try:
        async for message in ws:
            if isinstance(message, str):
                if message == "START_RECORDING":
                    print("[Server] >> START_RECORDING received. Beginning stream...")
                    is_recording = True
                elif message == "STOP_RECORDING":
                    print("[Server] >> STOP_RECORDING received. Pausing stream...")
                    is_recording = False
                elif message in ["STATE_NORMAL", "STATE_ABNORMAL", "STATE_PROCESSING"]:
                    print(f"[Server Status] >> {message}")
                else:
                    print(f"[Server] >> {message}")
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed.")

async def main(args):
    global is_recording
    is_recording = False
    
    print("Verifying audio format...")
    audio_data = verify_wav_format(args.file)
    print(f"Loaded audio successfully. Total bytes: {len(audio_data)} (~{len(audio_data)/(16000*2):.1f} seconds)")

    url = args.url
    if '?' not in url:
        url += "?device=test_wav"
        
    print(f"Connecting to WebSocket: {url}")
    
    try:
        async with websockets.connect(url) as ws:
            print("WebSocket connected successfully!")
            
            if args.auto_start:
                print("Auto-starting recording via API...")
                try:
                    # Parse host/port from ws URL to call HTTP API
                    import urllib.parse
                    parsed = urllib.parse.urlparse(args.url)
                    http_url = f"http://{parsed.netloc}/api/record/start"
                    res = requests.post(http_url)
                    print(f"API Response: {res.json()}")
                except Exception as e:
                    print(f"Failed to auto-start: {e}")

            stream_task = asyncio.create_task(stream_audio(ws, audio_data, args.loop))
            receive_task = asyncio.create_task(receive_messages(ws))
            
            await asyncio.gather(stream_task, receive_task)
            
    except ConnectionRefusedError:
        print("Error: Could not connect to the server. Is it running?")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream WAV file via WebSockets like ESP32")
    parser.add_argument("--file", required=True, help="Path to 16kHz mono int16 WAV file")
    parser.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL")
    parser.add_argument("--loop", action="store_true", help="Loop the audio file continuously")
    parser.add_argument("--auto-start", action="store_true", help="Automatically trigger /api/record/start on connect")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
