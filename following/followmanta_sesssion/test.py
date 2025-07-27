# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import serial
import time

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 115200
TIMEOUT = 0.1  # seconds
DELAY = 2  # seconds
DATA = '[1,1,1,1,1,1]\r\n'

# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
time.sleep(DELAY)  # Wait for serial port to initialize

try:
    while True:
        ser.write(DATA.encode('utf-8'))
        print(DATA)
        time.sleep(DELAY)
except KeyboardInterrupt:
    print('Program terminated by user.')
finally:
    ser.close()