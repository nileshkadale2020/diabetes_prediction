#!/usr/bin/env bash
# Monitor the training PID and, when it finishes, restart Flask and run a sample /predict
PID=65574
TRAIN_LOG=/tmp/model_training.log
APP_LOG=/tmp/diab_app.log
OUT=/tmp/post_predict_response.json
STATUS=/tmp/training_monitor_status.log

echo "[watcher] started at $(date)" >> "$STATUS"

echo "[watcher] waiting for PID $PID to exit..." >> "$STATUS"
# Wait until the PID no longer exists
while kill -0 "$PID" 2>/dev/null; do
  sleep 10
done

echo "[watcher] detected training process $PID finished at $(date)" >> "$STATUS"

echo "[watcher] app restart: stopping any existing app..." >> "$STATUS"
pkill -f 'python app.py' || true
sleep 1

echo "[watcher] starting Flask app with current environment..." >> "$STATUS"
# Start Flask app (assumes the current shell's python uses the diab-py311 env via conda run)
# We'll use nohup so this watcher can exit while Flask remains running
nohup python -u app.py > "$APP_LOG" 2>&1 &
APP_PID=$!

echo "[watcher] Flask started with PID $APP_PID" >> "$STATUS"
# give the server a moment to boot
sleep 3

echo "[watcher] sending sample POST to /predict" >> "$STATUS"
curl -s -X POST http://127.0.0.1:5000/predict \
  -F pregnancies=6 -F glucose=148 -F blood_pressure=72 -F skin_thickness=35 \
  -F insulin=0 -F bmi=33.6 -F diabetes_pedigree=0.627 -F age=50 \
  -H "Accept: application/json" -w "\nHTTP_CODE:%{http_code}\n" > "$OUT"

echo "[watcher] sample POST written to $OUT" >> "$STATUS"
echo "[watcher] finished at $(date)" >> "$STATUS"
