import cv2
import argparse

def main(server_url):
    cap = cv2.VideoCapture(server_url)

    if not cap.isOpened():
        print(f"Failed to connect to stream at {server_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam Stream", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://192.168.4.255:8080/video',
                        help='URL of the webcam stream server')
    args = parser.parse_args()

    main(args.url)