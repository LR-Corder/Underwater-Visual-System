import serial
import time

# 初始化串口通信
ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.1)  # 修改串口端口和波特率
while True:
    time.sleep(2) # 等待串口初始化

    serial_data=('[' + '1' + ','+ '1' +','+'1' +',' + '1'  + ','+ '1' +','+'1' +']\r\n')
                    
    # if send_flag:
    print(serial_data)
    # print("kaiqi",start_received)
    # if start_received
    ser.write(serial_data.encode('utf-8'))