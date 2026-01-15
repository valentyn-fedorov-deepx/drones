from flask import Flask, Response
import cv2
import atexit

app = Flask(__name__)
camera = cv2.VideoCapture(2)

if not camera.isOpened():
    raise RuntimeError("Failed to open webcam")

@atexit.register
def cleanup():
    print("Releasing camera")
    camera.release()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)