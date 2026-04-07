/*  Bridge Health Monitor — ESP32 Inference Sketch  (SCALER-FIXED VERSION)
    =============================================================================
    Hardware : ESP32 Dev Module + MPU-6050
    Features :
      1. Reads 128 samples from MPU-6050 (~1.28 s of data)
      2. Computes FFT on ax / ay / az  →  96 frequency features
      3. Applies StandardScaler z-score normalisation (from embedded scaler_params.json)
      4. Runs TFLite int8 model on-chip (no cloud needed for inference)
      5. On anomaly (score > 0.5) → publishes MQTT alert to broker.hivemq.com
      6. Publishes live sensor readings every 2 seconds for the dashboard
      7. After 3 consecutive anomaly windows → fires a HIGH-PRIORITY alert

    Libraries (install via Arduino Library Manager):
      - Adafruit MPU6050
      - Adafruit Unified Sensor
      - TensorFlowLite_ESP32
      - ArduinoFFT
      - PubSubClient
      - ArduinoJson

    Files needed in same sketch folder:
      bridge_monitor.ino   ← this file
      bridge_model1.h      ← generated with: xxd -i bridge_model.tflite > bridge_model1.h

    FIX APPLIED:
      Embedded scaler_params.json (96 mean + 96 scale values) directly into
      the sketch. Features are now z-score normalised BEFORE quantisation,
      which resolves the "Score: 0.000" scaler-mismatch issue.
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <arduinoFFT.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// TFLite Micro
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "bridge_model1.h"   // ← your model byte array

// ─────────────────────────────────────────────
//  CONFIGURATION  —  edit these
// ─────────────────────────────────────────────

const char* WIFI_SSID     = "12023002002099";
const char* WIFI_PASSWORD = "uem@kolkata";

const char* MQTT_BROKER  = "broker.hivemq.com";
const int   MQTT_PORT    = 1883;

const char* TOPIC_DATA   = "bridge/sensor/data";
const char* TOPIC_ALERT  = "bridge/sensor/alert";
const char* TOPIC_STATUS = "bridge/status";

const char* DEVICE_ID    = "bridge-node-01";

// ─────────────────────────────────────────────
//  SIGNAL / ML CONSTANTS
// ─────────────────────────────────────────────

#define SAMPLE_RATE        100
#define WINDOW_SIZE        128
#define N_FFT_BINS          32
#define FEATURE_SIZE        96   // 3 axes × 32 bins

#define ANOMALY_THRESHOLD  0.5f
#define ALERT_CONSECUTIVE    3   // fire alert after N consecutive anomaly windows

// ─────────────────────────────────────────────
//  EMBEDDED SCALER PARAMS  (from scaler_params.json)
//  96 mean values followed by 96 scale values
//  Applied as: z = (x - mean) / scale
// ─────────────────────────────────────────────

const float SCALER_MEAN[FEATURE_SIZE] = {
    // ax bins [0..31]
    3.4981266969786002f,  3.7011010253886405f,  5.8664909009299375f,  11.474974917545737f,
    12.180840675107833f,  11.35640193868055f,   8.94357666266562f,    5.738238893107007f,
    7.346472789335613f,   8.26440643743214f,    8.347610490241028f,   8.01839444775641f,
    7.868808225846376f,   6.18664265133725f,    4.069445001343732f,   3.402758734989778f,
    3.382410716096513f,   2.9273881708274025f,  2.712819911619573f,   2.5199599535624566f,
    1.5999175287418619f,  1.3462610584828194f,  1.2446879991119046f,  1.1587669450194f,
    1.1139043019262092f,  1.0481676471404073f,  0.9959295914855147f,  0.9972074667351911f,
    0.9509804147142286f,  0.9756590964379309f,  0.9368743613264512f,  0.9324230337404337f,
    // ay bins [32..63]
    1.8223130296028625f,  2.015075302991776f,   3.3129733047339185f,  7.0612022962803485f,
    7.751384976535573f,   7.295251368378291f,   5.892757699378102f,   3.583579611371056f,
    3.797088642597981f,   4.288136440273686f,   4.391794681575745f,   4.110066844519396f,
    3.8868655462289197f,  2.769798567540819f,   1.5572902894976741f,  1.3601280284083064f,
    1.158606269970305f,   1.1102113623865535f,  1.0308999885340402f,  1.0084094461887132f,
    0.9648560283990101f,  0.9130418411520508f,  0.9084694824494901f,  0.8900755092608331f,
    0.83825694680045f,    0.8624804311327476f,  0.8212425806304009f,  0.8392229207984765f,
    0.8186870099274277f,  0.868362706145694f,   0.8333531382798188f,  0.8484923893581355f,
    // az bins [64..95]
    128.01507551971653f,  1.2023781570728183f,  1.2368490091816196f,  1.2287852079485297f,
    1.2280140986559391f,  1.2218801150174017f,  1.2849641266881282f,  1.2418753206007502f,
    1.234634230197863f,   1.249089066574888f,   1.2452147467315269f,  1.2800251499778617f,
    1.2808468283714198f,  1.2320551226906888f,  1.2625242052788719f,  1.2724131955665166f,
    1.2594176889274955f,  1.2892518701178504f,  1.2657182716135478f,  1.2548979758140508f,
    1.2142158324490158f,  1.2677458431705086f,  1.2484563379981921f,  1.257341342429166f,
    1.2268334376739076f,  1.2591540930183662f,  1.2032416399061538f,  1.2657686170137228f,
    1.2722204932565122f,  1.242864380561722f,   1.2034401833135324f,  1.2411250057530219f
};

const float SCALER_SCALE[FEATURE_SIZE] = {
    // ax bins [0..31]
    3.363884571495276f,   3.215570653693232f,   6.566855475825664f,   13.449886964798912f,
    13.12077860194257f,   12.638926222348408f,  11.463671366700845f,  5.294895957298799f,
    7.411459232096286f,   7.347647971511269f,   7.472117794546635f,   7.308889029146001f,
    7.537043407766938f,   6.357639546349218f,   4.3407041232611565f,  3.7757240663961897f,
    4.3513497351820005f,  3.838637635842307f,   3.793204297922881f,   3.9070105581317067f,
    1.506183246016245f,   0.8999869062969444f,  0.7618279437839006f,  0.7017375778646103f,
    0.6788714554727991f,  0.6599720242592553f,  0.6228446049883142f,  0.5837018076322271f,
    0.5668993992080622f,  0.5759922328882925f,  0.5638667240919755f,  0.5436957976490865f,
    // ay bins [32..63]
    1.7788050292544006f,  1.537754045316881f,   3.7078839640915f,     8.539367436368599f,
    8.249126814100578f,   8.058361227330934f,   7.318270774930193f,   3.2690892423326905f,
    3.9307631980274818f,  4.054058627399232f,   4.264487188154053f,   4.070313841577184f,
    4.344470752309172f,   3.3344670254036974f,  0.996793147473969f,   0.7297201961647861f,
    0.6207567210821313f,  0.5656186327517578f,  0.5931234871927714f,  0.5835168433212204f,
    0.5700290706446959f,  0.5411631608950788f,  0.5494237209883551f,  0.5153317995651044f,
    0.538910098637638f,   0.5323521308517577f,  0.5118299386056995f,  0.5410277352938362f,
    0.5022402636655833f,  0.5292407669959055f,  0.5085142397933292f,  0.5158977465302697f,
    // az bins [64..95]
    1.4356423964586151f,  0.6509070915803479f,  0.6869542498557117f,  0.6690163586494852f,
    0.6937418820828719f,  0.6969954139625401f,  0.7205239426088861f,  0.6873461816654687f,
    0.7006821871320552f,  0.6914023798652875f,  0.6689225278149844f,  0.7042569940839273f,
    0.7299689070366853f,  0.7001392225993525f,  0.7086881012546835f,  0.7212071783751053f,
    0.692363572200648f,   0.7234919123976605f,  0.688327148774042f,   0.7083001765000961f,
    0.6766538640470645f,  0.7526301835765402f,  0.7227043631642175f,  0.6936162902262387f,
    0.7130862451411056f,  0.7054781985343282f,  0.6680190383746318f,  0.6934882063896189f,
    0.7193524653372162f,  0.6952204111652468f,  0.6779973042092967f,  0.6858012957670027f
};

// ─────────────────────────────────────────────
//  TENSOR ARENA
// ─────────────────────────────────────────────

constexpr int TENSOR_ARENA_SIZE = 20 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// ─────────────────────────────────────────────
//  GLOBAL OBJECTS
// ─────────────────────────────────────────────

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter*     error_reporter = &micro_error_reporter;

const tflite::Model*       tfl_model     = nullptr;
tflite::MicroInterpreter*  interpreter   = nullptr;
TfLiteTensor*              input_tensor  = nullptr;
TfLiteTensor*              output_tensor = nullptr;

Adafruit_MPU6050 mpu;
WiFiClient       wifiClient;
PubSubClient     mqttClient(wifiClient);
ArduinoFFT<double> FFT;

// ─────────────────────────────────────────────
//  SAMPLING BUFFERS
// ─────────────────────────────────────────────

double vReal_ax[WINDOW_SIZE], vImag_ax[WINDOW_SIZE];
double vReal_ay[WINDOW_SIZE], vImag_ay[WINDOW_SIZE];
double vReal_az[WINDOW_SIZE], vImag_az[WINDOW_SIZE];
float  features[FEATURE_SIZE];

// ─────────────────────────────────────────────
//  RUNTIME STATE
// ─────────────────────────────────────────────

unsigned long lastSampleTime  = 0;
unsigned long lastPublishTime = 0;
int  sampleIndex     = 0;
int  anomalyCount    = 0;
int  totalAlerts     = 0;
int  totalWindows    = 0;

// ─────────────────────────────────────────────
//  FORWARD DECLARATIONS
// ─────────────────────────────────────────────

void initMPU6050();
void initWiFi();
void initMQTT();
void reconnectMQTT();
void initTFLite();
void collectSample(int idx);
void runInference();
void publishLiveData();
void publishAlert(float score);


// =============================================================
//  SETUP
// =============================================================

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n================================================");
    Serial.println("       Bridge Health Monitor — ESP32 Node");
    Serial.println("       (Scaler-Fixed Version)");
    Serial.println("================================================\n");

    initMPU6050();
    initWiFi();
    initMQTT();
    initTFLite();

    Serial.println("\n[READY] Monitoring bridge vibrations...\n");
}


// =============================================================
//  MAIN LOOP
// =============================================================

void loop() {

    // Keep MQTT alive
    if (!mqttClient.connected())
        reconnectMQTT();
    mqttClient.loop();

    // ── Sample at SAMPLE_RATE Hz ──────────────────────────────
    if (millis() - lastSampleTime >= (1000 / SAMPLE_RATE)) {
        lastSampleTime = millis();
        collectSample(sampleIndex);
        sampleIndex++;

        if (sampleIndex >= WINDOW_SIZE) {
            sampleIndex = 0;
            runInference();
        }
    }

    // ── Publish live sensor data every 2 s ───────────────────
    if (millis() - lastPublishTime >= 2000) {
        lastPublishTime = millis();
        publishLiveData();
    }
}


// =============================================================
//  COLLECT ONE ACCELEROMETER SAMPLE
// =============================================================

void collectSample(int idx) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    vReal_ax[idx] = a.acceleration.x;
    vReal_ay[idx] = a.acceleration.y;
    vReal_az[idx] = a.acceleration.z;

    vImag_ax[idx] = vImag_ay[idx] = vImag_az[idx] = 0.0;
}


// =============================================================
//  RUN INFERENCE  (FFT → z-score normalise → TFLite → alert)
// =============================================================

void runInference() {

    totalWindows++;

    // ── Step 1: FFT on each axis ──────────────────────────────

    FFT.windowing(vReal_ax, WINDOW_SIZE, FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute   (vReal_ax, vImag_ax, WINDOW_SIZE, FFTDirection::Forward);
    FFT.complexToMagnitude(vReal_ax, vImag_ax, WINDOW_SIZE);

    FFT.windowing(vReal_ay, WINDOW_SIZE, FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute   (vReal_ay, vImag_ay, WINDOW_SIZE, FFTDirection::Forward);
    FFT.complexToMagnitude(vReal_ay, vImag_ay, WINDOW_SIZE);

    FFT.windowing(vReal_az, WINDOW_SIZE, FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute   (vReal_az, vImag_az, WINDOW_SIZE, FFTDirection::Forward);
    FFT.complexToMagnitude(vReal_az, vImag_az, WINDOW_SIZE);

    // ── Step 2: Build feature vector [FFT_ax | FFT_ay | FFT_az] ──

    for (int i = 0; i < N_FFT_BINS; i++) {
        features[i]              = (float)vReal_ax[i];
        features[N_FFT_BINS + i] = (float)vReal_ay[i];
        features[2*N_FFT_BINS+i] = (float)vReal_az[i];
    }

    // ── Step 3: Z-score normalise using embedded scaler params ──
    //    z = (x - mean) / scale   — matches Python StandardScaler
    //    This is the fix for the "Score: 0.000" scaler mismatch.

    for (int i = 0; i < FEATURE_SIZE; i++) {
        features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }

    // ── Step 4: Quantise normalised features into int8 ───────

    float   in_scale = input_tensor->params.scale;
    int32_t in_zp    = input_tensor->params.zero_point;

    for (int i = 0; i < FEATURE_SIZE; i++) {
        int val = (int)roundf(features[i] / in_scale) + in_zp;
        val = constrain(val, -128, 127);
        input_tensor->data.int8[i] = (int8_t)val;
    }

    // ── Step 5: Run the model ─────────────────────────────────

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[ERROR] TFLite inference failed!");
        return;
    }

    // ── Step 6: Decode output ─────────────────────────────────

    float   out_scale = output_tensor->params.scale;
    int32_t out_zp    = output_tensor->params.zero_point;
    float   score     = (output_tensor->data.int8[0] - out_zp) * out_scale;

    bool isAnomaly = (score > ANOMALY_THRESHOLD);

    // ── Step 7: Print to Serial Monitor ──────────────────────

    Serial.printf("[Window %4d]  Score: %.3f  →  %s\n",
                  totalWindows, score,
                  isAnomaly ? "*** ANOMALY ***" : "Normal");

    // ── Step 8: Consecutive-anomaly alert logic ───────────────

    if (isAnomaly) {
        anomalyCount++;
        Serial.printf("              Consecutive anomaly count: %d / %d\n",
                      anomalyCount, ALERT_CONSECUTIVE);

        if (anomalyCount >= ALERT_CONSECUTIVE) {
            totalAlerts++;
            publishAlert(score);
            anomalyCount = 0;
            Serial.printf("              ALERT #%d FIRED!\n", totalAlerts);
        }
    } else {
        anomalyCount = 0;
    }
}


// =============================================================
//  PUBLISH LIVE SENSOR DATA  (every 2 s)
// =============================================================

void publishLiveData() {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    StaticJsonDocument<256> doc;
    doc["device"]    = DEVICE_ID;
    doc["timestamp"] = millis();
    doc["ax"]        = a.acceleration.x;
    doc["ay"]        = a.acceleration.y;
    doc["az"]        = a.acceleration.z;
    doc["temp_c"]    = temp.temperature;

    char payload[256];
    serializeJson(doc, payload);

    if (mqttClient.publish(TOPIC_DATA, payload)) {
        Serial.printf("[MQTT] Live data published (ax=%.3f)\n",
                      a.acceleration.x);
    }
}


// =============================================================
//  PUBLISH ANOMALY ALERT
// =============================================================

void publishAlert(float score) {
    StaticJsonDocument<256> doc;
    doc["device"]   = DEVICE_ID;
    doc["alert"]    = "STRUCTURAL_ANOMALY_DETECTED";
    doc["score"]    = score;
    doc["severity"] = (score > 0.85f) ? "HIGH" : "MEDIUM";
    doc["message"]  = "Unusual vibration pattern detected. Inspection recommended.";
    doc["windows"]  = totalWindows;

    char payload[256];
    serializeJson(doc, payload);

    if (mqttClient.publish(TOPIC_ALERT, payload, true)) {
        Serial.println("[MQTT] *** ALERT PUBLISHED ***");
        Serial.printf("       Score: %.3f  Severity: %s\n",
                      score, score > 0.85f ? "HIGH" : "MEDIUM");
    } else {
        Serial.println("[MQTT] Alert publish FAILED — check connection");
    }
}


// =============================================================
//  INIT HELPERS
// =============================================================

void initMPU6050() {
    Serial.print("[INIT] MPU-6050... ");
    if (!mpu.begin()) {
        Serial.println("FAILED! Check wiring (SDA→GPIO21, SCL→GPIO22)");
        while (1) delay(10);
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("OK");
}

void initWiFi() {
    Serial.printf("[INIT] WiFi → %s", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 30) {
        delay(500); Serial.print("."); tries++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf(" OK  IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println(" FAILED — running in offline mode");
    }
}

void initMQTT() {
    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setBufferSize(512);
    reconnectMQTT();
}

void reconnectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return;
    int tries = 0;
    while (!mqttClient.connected() && tries < 5) {
        Serial.print("[MQTT] Connecting...");
        String id = String("esp32-bridge-") + String(random(0xffff), HEX);
        if (mqttClient.connect(id.c_str())) {
            Serial.println(" connected.");
            mqttClient.publish(TOPIC_STATUS,
                "{\"status\":\"online\",\"device\":\"bridge-node-01\"}");
        } else {
            Serial.printf(" failed (rc=%d). Retrying...\n", mqttClient.state());
            delay(2000); tries++;
        }
    }
}

void initTFLite() {
    Serial.print("[INIT] TFLite model... ");
    tfl_model = tflite::GetModel(bridge_model_tflite);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("FAILED: schema version mismatch!");
        while (1) delay(10);
    }
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE,
        error_reporter, nullptr, nullptr);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("FAILED: tensor allocation!");
        while (1) delay(10);
    }
    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("OK  (%d bytes of %d KB arena used)\n",
                  interpreter->arena_used_bytes(),
                  TENSOR_ARENA_SIZE / 1024);
}
