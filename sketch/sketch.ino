#include <Arduino_RouterBridge.h>
#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT22

// MQ-3 (alcohol/ethanol vapor) and MQ-135 (general air quality: NH3, benzene,
// smoke, other VOCs) gas sensors. Both modules expose an analog output (AO),
// used here for full-resolution readings, and a digital output (DO) driven by
// an onboard threshold potentiometer, which we don't use since AO already
// gives strictly more information than a single fixed trip point.
#define MQ3_PIN A0
#define MQ135_PIN A1

DHT dht(DHTPIN, DHTTYPE);
unsigned long lastRead = 0;
const long interval = 2000; // DHT22 max sample rate is ~2s

void setup() {
  Bridge.begin();
  dht.begin();

  // MQ-3/MQ-135 heater elements need to warm up before readings stabilize.
  // 20s is a practical minimum for a demo; accuracy keeps improving for the
  // first few minutes of operation.
  delay(20000);
}

void loop() {
  unsigned long now = millis();
  if (now - lastRead >= interval) {
    lastRead = now;
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    int alcohol_level = analogRead(MQ3_PIN);
    int air_quality_level = analogRead(MQ135_PIN);

    if (!isnan(h) && !isnan(t)) {
      Bridge.notify("record_reading", t, h, alcohol_level, air_quality_level);
    }
  }
}
