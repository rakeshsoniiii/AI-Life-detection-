#include <WiFi.h>
#include <WebServer.h>

// ===== WIFI SETTINGS =====
const char* ssid     = "A52s";      
const char* password = "00000000"; 

WebServer server(80);

// ===== MOTOR PINS =====
#define L_RPWM  25
#define L_LPWM  26
#define L_REN   27
#define L_LEN   14

#define R_RPWM  19
#define R_LPWM  32
#define R_REN   13
#define R_LEN   15

// ===== ULTRASONIC PINS =====
#define TRIG_PIN 4
#define ECHO_PIN 18

int currentL = 0;
int currentR = 0;

// ===== MOTOR CONTROL =====
void setMotors(int speedL, int speedR) {
  // Constrain limits just to be safe
  speedL = constrain(speedL, -255, 255);
  speedR = constrain(speedR, -255, 255);

  // Left Motor
  if (speedL >= 0) {
    analogWrite(L_RPWM, speedL);
    analogWrite(L_LPWM, 0);
  } else {
    analogWrite(L_RPWM, 0);
    analogWrite(L_LPWM, -speedL);
  }

  // Right Motor
  if (speedR >= 0) {
    analogWrite(R_RPWM, speedR);
    analogWrite(R_LPWM, 0);
  } else {
    analogWrite(R_RPWM, 0);
    analogWrite(R_LPWM, -speedR);
  }
}

void stopAll() {
  setMotors(0, 0);
  currentL = 0;
  currentR = 0;
}

// ===== SENSOR FUNCTION =====
int getDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000); 
  if (duration == 0) return 999; 
  return duration * 0.034 / 2;
}

// ===== ROUTE HANDLER =====
void handleDrive() {
  // MUST HAVE CORS so your separate HTML file can communicate with it
  server.sendHeader("Access-Control-Allow-Origin", "*");

  if (server.hasArg("L") && server.hasArg("R")) {
    currentL = server.arg("L").toInt();
    currentR = server.arg("R").toInt();
    setMotors(currentL, currentR);
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Missing L or R");
  }
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(L_RPWM, OUTPUT); pinMode(L_LPWM, OUTPUT);
  pinMode(L_REN,  OUTPUT); pinMode(L_LEN,  OUTPUT);
  digitalWrite(L_REN, HIGH); digitalWrite(L_LEN, HIGH);

  pinMode(R_RPWM, OUTPUT); pinMode(R_LPWM, OUTPUT);
  pinMode(R_REN,  OUTPUT); pinMode(R_LEN,  OUTPUT);
  digitalWrite(R_REN, HIGH); digitalWrite(R_LEN, HIGH);

  stopAll();

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("ESP32 IP ADDRESS: ");
  Serial.println(WiFi.localIP());

  server.on("/drive", handleDrive);
  server.begin();
}

// ===== LOOP =====
unsigned long lastSonicCheck = 0;

void loop() {
  server.handleClient();

  // Auto-reverse if obstacle detected while moving forward
  if (millis() - lastSonicCheck > 50) {
    lastSonicCheck = millis();
    int distance = getDistance();

    if (distance > 0 && distance < 13 && (currentL > 0 || currentR > 0)) {
      Serial.println("Obstacle! Auto-reversing...");
      stopAll();
      delay(50); 
      setMotors(-200, -200); // reverse both motors
      delay(300); 
      stopAll();
    }
  }
}
