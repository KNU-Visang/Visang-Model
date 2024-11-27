import RPi.GPIO as GPIO
import time

buzzer = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setwarnings(False)

pwm = GPIO.PWM(buzzer, 1.0)
pwm.start(1)

freq = 0

try:
	while(True):
		pwm.ChangeFrequency(freq+300)
		freq = (freq + 1) % 1000
		time.sleep(0.01)
except:
	pass

	
pwm.ChangeDutyCycle(0.0)
pwm.stop()
GPIO.cleanup()
