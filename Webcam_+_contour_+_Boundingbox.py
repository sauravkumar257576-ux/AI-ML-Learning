import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower = np.array([35, 50, 50])
upper = np.array([85, 255, 255])

kernel = np.ones(
    (3, 3),
    np.uint8
)

while True:

    # Camera se frame
    ret, frame = cap.read()

    if not ret:
        break

    # BGR → HSV
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # Green mask
    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # Noise cleaning
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Contours
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Every contour
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()