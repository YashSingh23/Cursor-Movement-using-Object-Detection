import cv2
import numpy as np
capture = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(50, 200, True)
frameCount = 0
while(1):
	ret, frame = capture.read()
	if not ret:
		break
	frameCount += 1
	resizedFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
	fgmask = fgbg.apply(resizedFrame)
	count = np.count_nonzero(fgmask)
	print('ok')
	if (frameCount > 1 and count > 1000):
		print('Someones stealing your honey')
		cv2.putText(resizedFrame, 'Someones stealing your honey', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	cv2.imshow('Frame', resizedFrame)
	cv2.imshow('Mask', fgmask)
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
capture.release()
cv2.destroyAllWindows()