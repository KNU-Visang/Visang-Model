import time
import subprocess

from picamera2 import Picamera2, Preview

from EyeBlinkDetector import EyeBlinkDetector

eyeBlinkDetector = EyeBlinkDetector()

picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(preview_config)

picam2.start_preview(Preview.QTGL)

picam2.start()

isBlink = False
BlinkTime = 0

musicScript = "Warn.py"
isScriptOn = False
process = None

while True:
	picam2.capture_file("eye.jpg")
	isBlinkList = eyeBlinkDetector.eye_blink_detector("eye.jpg")
	if len(isBlinkList) != 0:
		if isBlinkList[0][0] and isBlinkList[0][1]:
			print(isBlink, BlinkTime, time.time() - BlinkTime)
			if not isBlink:
				isBlink = True
				BlinkTime = time.time()
			else:
				if time.time() - BlinkTime > 3:
					print("Wake Up!!", time.time() - BlinkTime)
					if isScriptOn == False:
						isScriptOn = True
						process = subprocess.Popen(["python", musicScript])
		else:
			if isScriptOn:
				process.terminate()
				process.wait()
			isBlink = False
			isScriptOn = False

picam2.close()
