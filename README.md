# Bridge-Health-Monitoring-System
This system is organized as a three-stage pipeline: (1) data acquisition from the MPU-6050 via I2C, (2) on-chip signal processing and TFLite inference on the ESP32, and (3) MQTT-based alerting to a cloud broker monitored by a live browser dashboard.
