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
bool cancelMacro = false; 
unsigned long lastSonicCheck = 0;

// ===== MOTOR CONTROL =====
void setMotors(int speedL, int speedR) {
  currentL = constrain(speedL, -255, 255);
  currentR = constrain(speedR, -255, 255);

  if (currentL >= 0) {
    analogWrite(L_RPWM, currentL); analogWrite(L_LPWM, 0);
  } else {
    analogWrite(L_RPWM, 0); analogWrite(L_LPWM, -currentL);
  }

  if (currentR >= 0) {
    analogWrite(R_RPWM, currentR); analogWrite(R_LPWM, 0);
  } else {
    analogWrite(R_RPWM, 0); analogWrite(R_LPWM, -currentR);
  }
}

void stopAll() {
  setMotors(0, 0);
  cancelMacro = true; 
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

// ===== SMART DELAY (ANTI-CRASH ENGINE) =====
void smartDelay(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    server.handleClient(); // Keep listening to the app
    if (cancelMacro) break; // Stop if user pressed a button

    // Check Ultrasonic while doing the move
    if (millis() - lastSonicCheck > 50) {
      lastSonicCheck = millis();
      int dist = getDistance();
      
      // If obstacle is < 13cm and car is moving forward/turning
      if (dist > 0 && dist < 13 && (currentL > 0 || currentR > 0)) {
        Serial.print("CRASH PREVENTED! Distance: "); Serial.println(dist);
        cancelMacro = true;
        
        // Emergency Reverse
        setMotors(-200, -200); 
        delay(300); // Standard blocking delay is okay here just for the kickback
        setMotors(0, 0);
        break;
      }
    }
    delay(1);
  }
}

// ===== ROUTE HANDLERS =====
void handleDrive() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  if (server.hasArg("L") && server.hasArg("R")) {
    int L = server.arg("L").toInt();
    int R = server.arg("R").toInt();
    
    // If user touches joystick, cancel any running macro immediately
    if (L != 0 || R != 0) cancelMacro = true; 
    
    setMotors(L, R);
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Error");
  }
}

void handleMacro() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  String action = server.arg("action");
  Serial.println("Macro Triggered: " + action);
  
  cancelMacro = false; // Reset flag to allow move
  server.send(200, "text/plain", "Running"); 

  // --- THE SPECIAL MOVES ---
  // You can adjust the numbers inside smartDelay() to fine-tune the turns based on your motor power
  
  if (action == "stop") { stopAll(); }
  else if (action == "spin360") { setMotors(255, -255); smartDelay(1000); setMotors(0, 0); }
  else if (action == "uturn")   { setMotors(255, -255); smartDelay(500); setMotors(0, 0); }
  else if (action == "turn90R") { setMotors(255, -255); smartDelay(250); setMotors(0, 0); }
  else if (action == "turn90L") { setMotors(-255, 255); smartDelay(250); setMotors(0, 0); }
  else if (action == "fwd2")    { setMotors(255, 255); smartDelay(2000); setMotors(0, 0); }
  else if (action == "rev2")    { setMotors(-255, -255); smartDelay(2000); setMotors(0, 0); }
  else if (action == "drift")   { setMotors(255, 255); smartDelay(700); setMotors(255, -255); smartDelay(400); setMotors(0, 0); }
  else if (action == "circle")  { setMotors(255, 100); smartDelay(3000); setMotors(0, 0); }
  else if (action == "figure8") { setMotors(255, 80); smartDelay(2000); if(!cancelMacro) { setMotors(80, 255); smartDelay(2000); } setMotors(0, 0); }
  else if (action == "zigzag")  { for (int i=0; i<4; i++) { if(cancelMacro) break; setMotors(255, 120); smartDelay(400); setMotors(120, 255); smartDelay(400); } setMotors(0, 0); }
  else if (action == "snake")   { for (int i=0; i<3; i++) { if(cancelMacro) break; setMotors(200, 150); smartDelay(700); setMotors(150, 200); smartDelay(700); } setMotors(0, 0); }
  else if (action == "square")  { for (int i=0; i<4; i++) { if(cancelMacro) break; setMotors(200, 200); smartDelay(800); setMotors(255, -255); smartDelay(250); } setMotors(0, 0); }
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT); pinMode(ECHO_PIN, INPUT);
  
  pinMode(L_RPWM, OUTPUT); pinMode(L_LPWM, OUTPUT);
  pinMode(L_REN,  OUTPUT); pinMode(L_LEN,  OUTPUT);
  digitalWrite(L_REN, HIGH); digitalWrite(L_LEN, HIGH);

  pinMode(R_RPWM, OUTPUT); pinMode(R_LPWM, OUTPUT);
  pinMode(R_REN,  OUTPUT); pinMode(R_LEN,  OUTPUT);
  digitalWrite(R_REN, HIGH); digitalWrite(R_LEN, HIGH);
  stopAll();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  Serial.println("\nWiFi Connected. IP:");
  Serial.println(WiFi.localIP());

  server.on("/drive", handleDrive);
  server.on("/macro", handleMacro);
  server.begin();
}

// ===== LOOP =====
void loop() {
  server.handleClient();
  
  // Continuous safety check for manual joystick driving
  if (millis() - lastSonicCheck > 50) {
    lastSonicCheck = millis();
    int dist = getDistance();
    
    if (dist > 0 && dist < 13 && (currentL > 0 || currentR > 0)) {
      Serial.print("Manual Crash Prevented! Distance: "); Serial.println(dist);
      cancelMacro = true;
      setMotors(-200, -200);
      delay(300);
      setMotors(0, 0);
    }
  }
      }
