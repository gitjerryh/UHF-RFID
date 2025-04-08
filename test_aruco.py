#!/usr/bin/env python3
import cv2
import time
import numpy as np
from picamera2 import Picamera2

def main():
    # 初始化摄像头
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    # 等待摄像头初始化
    time.sleep(2)
    
    # 设置ArUco字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    
    print("开始测试ArUco标记检测...")
    print("将标记放在摄像头前，按Ctrl+C退出")
    
    try:
        while True:
            # 捕获图像
            frame = picam2.capture_array()
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 检测标记
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
            # 只输出检测结果
            if ids is not None:
                print(f"检测到 {len(ids)} 个标记!")
                for i, marker_id in enumerate(ids):
                    print(f"  - 标记 ID: {marker_id[0]}")
            else:
                print("未检测到标记")
            
            # 短暂暂停以降低CPU使用率
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n程序已停止")
    
    # 释放资源
    picam2.stop()
    print("测试结束")

if __name__ == "__main__":
    main()