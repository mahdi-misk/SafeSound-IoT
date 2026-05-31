#include <EasyWiFiMDNS.h>
#include <WebSocketsClient.h>

// ======================= Settings =======================
String websocket_server_host = ""; // Will be assigned automatically via mDNS
const uint16_t websocket_server_port = 8000;

EasyWiFiMDNS wifi;
WebSocketsClient webSocket;

// ======================= Pins =======================
#define MOTOR_IN1 12
#define MOTOR_IN2 14
#define LED_GREEN 27
#define LED_RED 26
#define BUZZER_PIN 25

// SPW2430 Mic Pin
// Important note: Must be an ADC1 pin (like 34, 35, 32, 33) because Wi-Fi
// disables ADC2 pins
#define MIC_PIN 34

// ======================= Audio Settings =======================
#define SAMPLE_RATE 8000
#define BUFFER_SIZE 1024

uint8_t bufferA[BUFFER_SIZE];
uint8_t bufferB[BUFFER_SIZE];
uint8_t *active_buffer = bufferA;
int buffer_index = 0;
unsigned long last_sample_time = 0;
const unsigned long sample_interval = 1000000 / SAMPLE_RATE; // 125 microseconds
bool is_recording = false;

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
  case WStype_DISCONNECTED:
    Serial.println("[WSc] Disconnected from Python Server!");
    break;
  case WStype_CONNECTED:
    Serial.printf("[WSc] Connected to url: %s\n", payload);
    webSocket.sendTXT("ESP32 Connected");
    break;
  case WStype_TEXT: {
    Serial.printf("[WSc] Command received: %s\n", payload);
    String text = String((char *)payload);

    if (text == "PUMP_ON") {
      digitalWrite(MOTOR_IN1, HIGH);
      digitalWrite(MOTOR_IN2, LOW);
    } else if (text == "PUMP_OFF") {
      digitalWrite(MOTOR_IN1, LOW);
      digitalWrite(MOTOR_IN2, LOW);
    } else if (text == "START_RECORDING") {
      is_recording = true;
      buffer_index = 0;
      Serial.println("[APP] Recording Started.");
    } else if (text == "STOP_RECORDING") {
      is_recording = false;
      digitalWrite(LED_GREEN, LOW);
      digitalWrite(LED_RED, LOW);
      digitalWrite(BUZZER_PIN, LOW);
      Serial.println("[APP] Recording Stopped.");
    } else if (text == "STATE_PROCESSING") {
      digitalWrite(LED_GREEN, HIGH);
      digitalWrite(LED_RED, HIGH);
      digitalWrite(BUZZER_PIN, LOW);
    } else if (text == "STATE_NORMAL") {
      digitalWrite(LED_GREEN, HIGH);
      digitalWrite(LED_RED, LOW);
      digitalWrite(BUZZER_PIN, LOW);
    } else if (text == "STATE_ABNORMAL") {
      digitalWrite(LED_GREEN, LOW);
      digitalWrite(LED_RED, HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
    }
  } break;
  }
}

void setup() {
  Serial.begin(115200);

  // Initialize output pins
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Initialize Mic pin as input
  pinMode(MIC_PIN, INPUT);

  // Turn everything off initially
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Connect to Wi-Fi using EasyWiFiMDNS
  wifi.setApPassword("12345678");
  wifi.setPortalTimeout(180);
  wifi.setConnectTimeout(20);
  wifi.setDebug(true);

  Serial.println("Connecting to WiFi via EasyWiFiMDNS...");
  if (wifi.begin("device_1")) {
    Serial.println("\nWiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(wifi.ip());
  } else {
    Serial.println("\nWiFi Connection failed or portal timed out!");
  }

  Serial.println("🔍 Searching for AI server (ai-server.local)...");
  websocket_server_host = MDNS.queryHost("ai-server").toString();

  if (websocket_server_host == "" || websocket_server_host == "0.0.0.0") {
    Serial.println("❌ Server not found! Using fallback IP...");
    websocket_server_host = "192.168.0.151"; // fallback IP
  } else {
    Serial.print("✅ Server found! IP: ");
    Serial.println(websocket_server_host);
  }

  // Connect to Python server
  webSocket.begin(websocket_server_host, websocket_server_port, "/ws");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);

  Serial.println("System Ready.");
}

void loop() {
  wifi.loop();
  webSocket.loop();

  if (is_recording) {
    unsigned long current_time = micros();

    // Read audio if it's time (every 125 microseconds for 8000Hz)
    if (current_time - last_sample_time >= sample_interval) {
      last_sample_time = current_time;

      // Read analog signal from mic (0 to 4095)
      int analog_val = analogRead(MIC_PIN);

      // Convert 12-bit value to 8-bit (0 to 255)
      active_buffer[buffer_index] = (analog_val >> 4) & 0xFF;
      buffer_index++;

      // If buffer is full, swap pointer and send the full one
      if (buffer_index >= BUFFER_SIZE) {
        uint8_t *send_buffer = active_buffer;

        // Immediate swap to the other array to avoid signal loss during
        // transmission
        if (active_buffer == bufferA) {
          active_buffer = bufferB;
        } else {
          active_buffer = bufferA;
        }
        buffer_index = 0;

        // Send full array over Wi-Fi
        webSocket.sendBIN(send_buffer, BUFFER_SIZE);
      }
    }
  }
}