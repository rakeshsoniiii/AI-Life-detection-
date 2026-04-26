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

int currentSpeed = 200; // default speed

// ===== MOTOR FUNCTIONS =====
void leftForward(int s)  { analogWrite(L_RPWM, s); analogWrite(L_LPWM, 0); }
void leftBackward(int s) { analogWrite(L_RPWM, 0); analogWrite(L_LPWM, s); }
void leftStop()          { analogWrite(L_RPWM, 0); analogWrite(L_LPWM, 0); }

void rightForward(int s)  { analogWrite(R_RPWM, s); analogWrite(R_LPWM, 0); }
void rightBackward(int s) { analogWrite(R_RPWM, 0); analogWrite(R_LPWM, s); }
void rightStop()          { analogWrite(R_RPWM, 0); analogWrite(R_LPWM, 0); }

void stopAll()      { leftStop();             rightStop();              }
void goForward()    { leftForward(currentSpeed);  rightForward(currentSpeed);  }
void goBackward()   { leftBackward(currentSpeed); rightBackward(currentSpeed); }
void goLeft()       { leftBackward(currentSpeed); rightForward(currentSpeed);  }
void goRight()      { leftForward(currentSpeed);  rightBackward(currentSpeed); }
void curveLeft()    { leftForward(currentSpeed/2); rightForward(currentSpeed); }
void curveRight()   { leftForward(currentSpeed);  rightForward(currentSpeed/2);}

// ===== HTML PAGE =====
String getHTML() {
  return R"rawhtml(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>RC Car</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
    body {
      background:#0f0f0f;
      color:white;
      font-family: Arial, sans-serif;
      height:100vh;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:space-between;
      padding:20px;
      overflow:hidden;
    }
    h2 { font-size:20px; color:#00e5ff; letter-spacing:2px; margin-top:5px; }
    .status {
      font-size:13px;
      color:#aaa;
      background:#1a1a1a;
      padding:6px 16px;
      border-radius:20px;
    }
    .status span { color:#00e5ff; font-weight:bold; }

    /* Speed slider */
    .speed-box {
      width:100%;
      background:#1a1a1a;
      border-radius:12px;
      padding:10px 16px;
      display:flex;
      align-items:center;
      gap:10px;
    }
    .speed-box label { font-size:13px; color:#aaa; white-space:nowrap; }
    .speed-box input[type=range] {
      flex:1;
      accent-color:#00e5ff;
      height:6px;
    }
    .speed-box span { font-size:14px; color:#00e5ff; font-weight:bold; min-width:36px; text-align:right; }

    /* D-PAD */
    .dpad {
      display:grid;
      grid-template-columns: repeat(3, 80px);
      grid-template-rows: repeat(3, 80px);
      gap:8px;
    }
    .btn {
      background:#1e1e1e;
      border: 2px solid #333;
      border-radius:14px;
      color:white;
      font-size:28px;
      display:flex;
      align-items:center;
      justify-content:center;
      cursor:pointer;
      user-select:none;
      transition: background 0.1s, transform 0.1s;
      -webkit-user-select:none;
    }
    .btn:active, .btn.pressed {
      background:#00e5ff;
      color:#000;
      border-color:#00e5ff;
      transform:scale(0.93);
    }
    .btn.stop {
      background:#1e1e1e;
      border-color:#ff3b3b;
      font-size:14px;
      color:#ff3b3b;
      font-weight:bold;
    }
    .btn.stop:active, .btn.stop.pressed {
      background:#ff3b3b;
      color:white;
    }
    .empty { visibility:hidden; }

    /* Bottom row */
    .bottom {
      display:flex;
      gap:12px;
      width:100%;
    }
    .btn-wide {
      flex:1;
      height:56px;
      background:#1e1e1e;
      border:2px solid #333;
      border-radius:14px;
      color:white;
      font-size:13px;
      font-weight:bold;
      display:flex;
      align-items:center;
      justify-content:center;
      gap:6px;
      cursor:pointer;
      user-select:none;
      -webkit-user-select:none;
      transition: background 0.1s;
    }
    .btn-wide:active, .btn-wide.pressed {
      background:#00e5ff;
      color:#000;
      border-color:#00e5ff;
    }
  </style>
</head>
<body>

  <h2>&#127950; RC CAR</h2>
  <div class="status">Status: <span id="statusText">STOP</span></div>

  <div class="speed-box">
    <label>Speed</label>
    <input type="range" id="speedSlider" min="80" max="255" value="200"
      oninput="updateSpeed(this.value)">
    <span id="speedVal">200</span>
  </div>

  <div class="dpad">
    <div class="btn empty"></div>
    <div class="btn" id="btn-F"
      ontouchstart="press('forward')" ontouchend="press('stop')"
      onmousedown="press('forward')"  onmouseup="press('stop')">&#9650;</div>
    <div class="btn empty"></div>

    <div class="btn" id="btn-L"
      ontouchstart="press('left')"    ontouchend="press('stop')"
      onmousedown="press('left')"     onmouseup="press('stop')">&#9664;</div>
    <div class="btn stop" id="btn-S"
      ontouchstart="press('stop')"    ontouchend="press('stop')"
      onmousedown="press('stop')"     onmouseup="press('stop')">STOP</div>
    <div class="btn" id="btn-R"
      ontouchstart="press('right')"   ontouchend="press('stop')"
      onmousedown="press('right')"    onmouseup="press('stop')">&#9654;</div>

    <div class="btn empty"></div>
    <div class="btn" id="btn-B"
      ontouchstart="press('backward')" ontouchend="press('stop')"
      onmousedown="press('backward')"  onmouseup="press('stop')">&#9660;</div>
    <div class="btn empty"></div>
  </div>

  <div class="bottom">
    <div class="btn-wide" id="btn-CL"
      ontouchstart="press('curveleft')"  ontouchend="press('stop')"
      onmousedown="press('curveleft')"   onmouseup="press('stop')">&#8630; Curve L</div>
    <div class="btn-wide" id="btn-CR"
      ontouchstart="press('curveright')" ontouchend="press('stop')"
      onmousedown="press('curveright')"  onmouseup="press('stop')">Curve R &#8631;</div>
  </div>

<script>
  const labels = {
    forward:'FORWARD', backward:'BACKWARD',
    left:'TURN LEFT', right:'TURN RIGHT',
    stop:'STOP', curveleft:'CURVE LEFT', curveright:'CURVE RIGHT'
  };

  const btnMap = {
    forward:'btn-F', backward:'btn-B',
    left:'btn-L', right:'btn-R',
    curveleft:'btn-CL', curveright:'btn-CR', stop:'btn-S'
  };

  let lastCmd = '';
  let sending = false;

  function press(cmd) {
    if (cmd === lastCmd) return;
    lastCmd = cmd;
    document.getElementById('statusText').innerText = labels[cmd] || cmd;

    // highlight button
    Object.values(btnMap).forEach(id => {
      document.getElementById(id)?.classList.remove('pressed');
    });
    if (btnMap[cmd]) document.getElementById(btnMap[cmd]).classList.add('pressed');

    sendCmd(cmd);
  }

  function sendCmd(cmd) {
    if (sending) return;
    sending = true;
    fetch('/cmd?action=' + cmd)
      .finally(() => { sending = false; });
  }

  function updateSpeed(val) {
    document.getElementById('speedVal').innerText = val;
    fetch('/speed?val=' + val);
  }

  // Prevent page scroll while using dpad
  document.addEventListener('touchmove', e => e.preventDefault(), {passive:false});
</script>

</body>
</html>
)rawhtml";
}

// ===== ROUTE HANDLERS =====
void handleRoot() {
  server.send(200, "text/html", getHTML());
}

void handleCmd() {
  String action = server.arg("action");
  Serial.println("CMD: " + action);

  if      (action == "forward")     goForward();
  else if (action == "backward")    goBackward();
  else if (action == "left")        goLeft();
  else if (action == "right")       goRight();
  else if (action == "curveleft")   curveLeft();
  else if (action == "curveright")  curveRight();
  else                              stopAll();

  server.send(200, "text/plain", "OK");
}

void handleSpeed() {
  currentSpeed = server.arg("val").toInt();
  currentSpeed = constrain(currentSpeed, 80, 255);
  Serial.println("Speed: " + String(currentSpeed));
  server.send(200, "text/plain", "OK");
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);

  // Motor 1 (Left)
  pinMode(L_RPWM, OUTPUT); pinMode(L_LPWM, OUTPUT);
  pinMode(L_REN,  OUTPUT); pinMode(L_LEN,  OUTPUT);
  digitalWrite(L_REN, HIGH); digitalWrite(L_LEN, HIGH);

  // Motor 2 (Right)
  pinMode(R_RPWM, OUTPUT); pinMode(R_LPWM, OUTPUT);
  pinMode(R_REN,  OUTPUT); pinMode(R_LEN,  OUTPUT);
  digitalWrite(R_REN, HIGH); digitalWrite(R_LEN, HIGH);

  stopAll();

  // Connect WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("Open this on your phone: http://");
  Serial.println(WiFi.localIP());

  // Routes
  server.on("/",      handleRoot);
  server.on("/cmd",   handleCmd);
  server.on("/speed", handleSpeed);
  server.begin();
  Serial.println("Web server started!");
}

// ===== LOOP =====
void loop() {
  server.handleClient();
}
