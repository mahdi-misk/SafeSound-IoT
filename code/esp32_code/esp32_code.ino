#include "driver/i2s.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebSocketsClient.h>

// ======================= Settings =======================
String websocket_server_host = "";
const uint16_t websocket_server_port = 8000;

const char* ssid = "SafeSound_AP";
const char* password = "password123";
WebSocketsClient webSocket;

// ======================= Pins =======================
// LED and Buzzer (moved from 25,26,27 to free them for I2S)
#define LED_GREEN 14
#define LED_RED 27
#define BUZZER_PIN 4

// INMP441 I2S Pins (clean pins - no boot/strapping issues)
#define I2S_WS 26
#define I2S_SCK 25
#define I2S_SD 33

// ======================= Audio Settings =======================
#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024
#define GAIN 1 // No amplification - INMP441 signal is strong enough after >>16
#define I2S_READ_SIZE 256 // Smaller I2S read chunks for smoother processing

int16_t bufferA[BUFFER_SIZE];
int16_t bufferB[BUFFER_SIZE];

volatile int16_t *active_buffer = bufferA;
volatile int buffer_index = 0;

volatile bool is_recording = false;
volatile bool buffer_ready = false;
int16_t *send_buffer_ptr = NULL;

portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

// DC offset tracking for INMP441
int32_t dc_offset = 0;
bool dc_initialized = false;

// Debug: track audio levels
unsigned long last_debug_print = 0;
int32_t max_sample_seen = 0;
int32_t min_sample_seen = 0;

// ======================= I2S Setup =======================
void setupI2S() {
  i2s_config_t i2s_config = {.mode =
                                 (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
                             .sample_rate = SAMPLE_RATE,
                             .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
                             .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
                             .communication_format = I2S_COMM_FORMAT_STAND_I2S,
                             .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
                             .dma_buf_count = 8,
                             .dma_buf_len = 512,
                             .use_apll = true,
                             .tx_desc_auto_clear = false,
                             .fixed_mclk = 0};

  i2s_pin_config_t pin_config = {.bck_io_num = I2S_SCK,
                                 .ws_io_num = I2S_WS,
                                 .data_out_num = I2S_PIN_NO_CHANGE,
                                 .data_in_num = I2S_SD};

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("[I2S] Driver install failed: %d\n", err);
    return;
  }

  err = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("[I2S] Pin config failed: %d\n", err);
    return;
  }

  i2s_zero_dma_buffer(I2S_NUM_0);

  // Discard initial garbage samples from INMP441 startup
  int32_t dummy[256];
  size_t dummyBytes;
  for (int i = 0; i < 10; i++) {
    i2s_read(I2S_NUM_0, dummy, sizeof(dummy), &dummyBytes, 100);
  }

  Serial.println("[I2S] INMP441 Ready (16kHz, APLL, gain=" + String(GAIN) +
                 "x).");
}

// ======================= Calibrate DC Offset =======================
void calibrateDCOffset() {
  Serial.println("[CAL] Calibrating DC offset...");
  int64_t sum = 0;
  int count = 0;
  int32_t samples[256];
  size_t bytesRead;

  for (int round = 0; round < 20; round++) {
    esp_err_t result =
        i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytesRead, 100);
    if (result == ESP_OK && bytesRead > 0) {
      int samplesRead = bytesRead / sizeof(int32_t);
      for (int i = 0; i < samplesRead; i++) {
        sum += (samples[i] >> 16); // Use upper 16 bits
        count++;
      }
    }
  }

  if (count > 0) {
    dc_offset = sum / count;
    dc_initialized = true;
    Serial.printf("[CAL] DC offset = %d (from %d samples)\n", (int)dc_offset,
                  count);
  } else {
    dc_offset = 0;
    dc_initialized = true;
    Serial.println("[CAL] Warning: Could not calibrate, using 0.");
  }
}

// ======================= WebSocket Event =======================
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
    String text = String((char *)payload);
    Serial.printf("[WSc] Command received: %s\n", payload);

    if (text == "START_RECORDING") {
      // Reset buffer state
      portENTER_CRITICAL(&mux);
      buffer_index = 0;
      buffer_ready = false;
      portEXIT_CRITICAL(&mux);

      // Re-calibrate DC offset before recording
      i2s_zero_dma_buffer(I2S_NUM_0);
      calibrateDCOffset();

      // Reset debug values
      max_sample_seen = 0;
      min_sample_seen = 0;

      is_recording = true;

      digitalWrite(LED_GREEN, HIGH);
      digitalWrite(LED_RED, LOW);
      digitalWrite(BUZZER_PIN, LOW);

      Serial.println("[APP] Recording Started (16kHz).");
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
      // is_recording remains true, so it continues streaming
    }
  } break;

  default:
    break;
  }
}

// ======================= Network Task Core 0 =======================
void networkTask(void *pvParameters) {
  for (;;) {
    webSocket.loop();

    if (buffer_ready) {
      portENTER_CRITICAL(&mux);
      int16_t *buf_to_send = send_buffer_ptr;
      buffer_ready = false;
      portEXIT_CRITICAL(&mux);

      if (buf_to_send != NULL && webSocket.isConnected()) {
        webSocket.sendBIN((uint8_t *)buf_to_send,
                          BUFFER_SIZE * sizeof(int16_t));
      }
    }

    vTaskDelay(2 / portTICK_PERIOD_MS);
  }
}

// ======================= Setup =======================
void setup() {
  Serial.begin(115200);

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  setupI2S();

  Serial.println("Starting WiFi Access Point...");
  WiFi.softAP(ssid, password);
  
  Serial.println("\nAccess Point Started!");
  Serial.print("ESP32 AP IP Address: ");
  Serial.println(WiFi.softAPIP());

  if (!MDNS.begin("device_1")) {
    Serial.println("Error setting up MDNS responder!");
  }

  Serial.println("Searching for AI server ai-server.local...");

  websocket_server_host = MDNS.queryHost("ai-server").toString();

  if (websocket_server_host == "" || websocket_server_host == "0.0.0.0") {
    Serial.println("Server not found! Using fallback IP...");
    websocket_server_host = "192.168.4.2";
  } else {
    Serial.print("Server found! IP: ");
    Serial.println(websocket_server_host);
  }

  webSocket.begin(websocket_server_host, websocket_server_port, "/ws");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);

  xTaskCreatePinnedToCore(networkTask, "NetworkTask", 8192, NULL, 1, NULL, 0);

  Serial.println("System Ready. INMP441 Audio on Core 1, WiFi on Core 0.");
}

// ======================= Loop Core 1 =======================
void loop() {
  if (!is_recording) {
    delay(10);
    return;
  }

  if (buffer_ready) {
    delay(1);
    return;
  }

  int32_t rawSamples[I2S_READ_SIZE];
  size_t bytesRead = 0;

  esp_err_t result = i2s_read(I2S_NUM_0, rawSamples, sizeof(rawSamples),
                              &bytesRead, portMAX_DELAY);

  if (result == ESP_OK && bytesRead > 0) {
    int samplesRead = bytesRead / sizeof(int32_t);

    // Process samples WITHOUT critical section (only lock when swapping
    // buffers)
    for (int i = 0; i < samplesRead && buffer_index < BUFFER_SIZE; i++) {
      int32_t sample32 = rawSamples[i];

      // INMP441: 24-bit data in upper bits of 32-bit word
      // Shift right by 16 to get a proper 16-bit signed value
      int32_t sample_shifted = sample32 >> 16;

      // Remove DC offset
      int32_t sample_clean = sample_shifted - dc_offset;

      // Apply software gain
      int32_t sample_amplified = sample_clean * GAIN;

      // Clamp to int16 range to prevent distortion
      if (sample_amplified > 32767)
        sample_amplified = 32767;
      if (sample_amplified < -32768)
        sample_amplified = -32768;

      int16_t sample16 = (int16_t)sample_amplified;

      // Track levels for debug
      if (sample16 > max_sample_seen)
        max_sample_seen = sample16;
      if (sample16 < min_sample_seen)
        min_sample_seen = sample16;

      ((int16_t *)active_buffer)[buffer_index++] = sample16;
    }

    // When buffer is full, swap buffers (only this part needs critical section)
    if (buffer_index >= BUFFER_SIZE) {
      portENTER_CRITICAL(&mux);
      send_buffer_ptr = (int16_t *)active_buffer;

      if (active_buffer == bufferA) {
        active_buffer = bufferB;
      } else {
        active_buffer = bufferA;
      }

      buffer_index = 0;
      buffer_ready = true;
      portEXIT_CRITICAL(&mux);

      // Print audio level debug info every 2 seconds
      unsigned long now = millis();
      if (now - last_debug_print > 2000) {
        last_debug_print = now;
        Serial.printf("[AUDIO] Level: min=%d max=%d\n", (int)min_sample_seen,
                      (int)max_sample_seen);
        max_sample_seen = 0;
        min_sample_seen = 0;
      }
    }
  }
}