import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=True)

contour_counts = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    fgmask = fgbg.apply(frame)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_count = 0  # Count the number of contours

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contour_count += 1

    contour_counts.append(contour_count)  # Add contour count to the list

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Data Visualization using Matplotlib
plt.plot(contour_counts)
plt.title('Contour Count over Time')
plt.xlabel('Frame Number')
plt.ylabel('Contour Count')
plt.show()