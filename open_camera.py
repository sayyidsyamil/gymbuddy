import cv2
from ultralytics import YOLO


MODEL_NAME = "yolo26n.pt"
CONFIDENCE = 0.35


def main():
    model = YOLO(MODEL_NAME)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera. Check camera permissions and try again.")
        return

    print("Object detection started. Press 'q' or Esc to close the window.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Could not read from camera.")
            break

        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("GymBuddy Object Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
