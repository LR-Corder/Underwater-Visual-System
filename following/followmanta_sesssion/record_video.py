import cv2  
import time
  
def record_video(save_path, duration=10):  
    # 打开摄像头  
    cap = cv2.VideoCapture(0)
  
    # 定义编解码器和创建VideoWriter对象  
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (1280,  480))  
  
    # 录制视频  
    count = 0  
    while(cap.isOpened()):  
        start = time.time()
        ret, frame = cap.read()  
        if ret:  
            # 如果需要，可以在这里处理frame  
            # frame = cv2.flip(frame, 0)  # 翻转帧，作为示例  
  
            # 写入帧到视频文件  
            # out.write(frame)  
  
            # 显示结果帧  
            # cv2.imshow('frame', frame)  
            pass
  
            # 按'q'退出循环  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
              
            # 如果设置了录制时长，则退出循环  
            # if count >= duration * 20:  # 假设帧率为20fps  
            #     break  
            # count += 1  
        else:  
            break  
        print('time---', time.time()-start)
  
    # 释放捕获和销毁所有窗口  
    cap.release()  
    out.release()  
    cv2.destroyAllWindows()  
  
# 调用函数录制视频，保存为output.avi，录制时长为10秒  
record_video('output.avi', 10)
