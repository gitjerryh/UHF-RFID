#!/usr/bin/env python3
import numpy as np
import json
import time
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import serial
import re
import cv2
from picamera2 import Picamera2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rfid_analysis.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TagData:
    """标签数据结构"""
    id: str
    rssi: int
    rssi_dbm: int
    distance: float
    is_stable: bool
    timestamp: int

class RFIDAnalyzer:
    """RFID数据分析器"""
    
    def __init__(self):
        # 配置参数 - 调整为更合理的值
        self.rssi_window_size = 10  # RSSI滑动窗口大小
        self.distance_threshold = 1.0  # 距离变化阈值(米) - 放宽到1米
        self.rssi_std_threshold = 8.0  # RSSI标准差阈值 - 放宽到8
        self.min_samples = 3  # 最小样本数 - 减少到3个
        self.vision_mode_duration = 10  # 视觉模式持续时间(秒)
        
        # 数据存储
        self.tag_history: Dict[str, List[TagData]] = {}
        self.reliability_reasons: Dict[str, List[str]] = {}
        
        # 状态
        self.current_mode = "RFID"  # RFID或VISION
        self.last_switch_time = time.time()
        
        # ArUco相关
        self.aruco_initialized = False
        self.aruco_markers_detected = False
        self.picam2 = None
        self.marker_size = 0.05  # ArUco标记的边长，单位：米
        self.camera_matrix = np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
    def add_data(self, data: dict) -> bool:
        """添加新的标签数据并分析可靠性"""
        try:
            # 创建TagData对象
            tag_data = TagData(
                id=data['id'],
                rssi=data['rssi'],
                rssi_dbm=data['rssiDbm'],
                distance=data['distance'],
                is_stable=data['isStable'],
                timestamp=data['timestamp']
            )
            
            # 更新标签历史
            if tag_data.id not in self.tag_history:
                self.tag_history[tag_data.id] = []
                self.reliability_reasons[tag_data.id] = []
            
            self.tag_history[tag_data.id].append(tag_data)
            
            # 保持历史数据在窗口大小内
            if len(self.tag_history[tag_data.id]) > self.rssi_window_size:
                self.tag_history[tag_data.id].pop(0)
            
            # 分析可靠性
            is_reliable, reasons = self._analyze_reliability(tag_data)
            self.reliability_reasons[tag_data.id] = reasons
            
            # 如果数据不可靠，立即切换到视觉模式
            if not is_reliable and self.current_mode != "VISION":
                self._switch_to_vision_mode(tag_data)
            
            # 检查是否需要从视觉模式返回RFID模式
            self._check_vision_mode_timeout()
            
            return is_reliable
            
        except Exception as e:
            logging.error(f"数据处理错误: {e}")
            return False
    
    def _analyze_reliability(self, tag_data: TagData) -> tuple:
        """分析标签数据的可靠性，返回(是否可靠, 原因列表)"""
        history = self.tag_history[tag_data.id]
        reasons = []
        
        # 0. 检查样本数量
        if len(history) < self.min_samples:
            reasons.append(f"样本数不足: {len(history)}/{self.min_samples}")
            return False, reasons
        
        # 1. 检查RSSI稳定性
        rssi_values = [d.rssi_dbm for d in history]
        rssi_std = np.std(rssi_values)
        if rssi_std > self.rssi_std_threshold:
            reasons.append(f"RSSI不稳定: 标准差={rssi_std:.2f}>{self.rssi_std_threshold}")
            logging.warning(f"标签 {tag_data.id} RSSI不稳定: 标准差={rssi_std:.2f}")
            return False, reasons
        
        # 2. 检查距离变化 - 只检查最近两次，不检查整个历史
        if len(history) > 1:
            last_distance = history[-2].distance
            distance_change = abs(tag_data.distance - last_distance)
            if distance_change > self.distance_threshold:
                reasons.append(f"距离变化过大: {distance_change:.2f}米>{self.distance_threshold}米")
                logging.warning(f"标签 {tag_data.id} 距离变化过大: {distance_change:.2f}米")
                return False, reasons
        
        # 3. 检查RSSI值范围 - 放宽范围限制
        if tag_data.rssi_dbm > -15 or tag_data.rssi_dbm < -110:
            reasons.append(f"RSSI值异常: {tag_data.rssi_dbm}dBm")
            logging.warning(f"标签 {tag_data.id} RSSI值异常: {tag_data.rssi_dbm}dBm")
            return False, reasons
        
        return True, ["数据可靠"]
    
    def _switch_to_vision_mode(self, tag_data: TagData):
        """切换到视觉定位模式"""
        if self.current_mode != "VISION":
            self.current_mode = "VISION"
            self.last_switch_time = time.time()
            logging.info(f"切换到视觉模式 - 标签: {tag_data.id}")
            
            # 初始化ArUco检测，检查初始化结果
            if not self._initialize_aruco_detection():
                logging.error("视觉模式初始化失败，返回RFID模式")
                self.current_mode = "RFID"
                return
            
            # 启动ArUco检测
            self.aruco_markers_detected = False
            self._run_aruco_detection()
    
    def _initialize_aruco_detection(self):
        """初始化ArUco标记检测"""
        try:
            # 确保先释放摄像头资源
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except:
                    pass
                self.picam2 = None
                self.aruco_initialized = False
                time.sleep(1)  # 等待资源完全释放
            
            # 检查是否有摄像头连接
            cameras = Picamera2.global_camera_info()
            if not cameras:
                logging.error("未检测到摄像头，请检查连接")
                return False
                
            # 重新初始化摄像头
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
            self.picam2.configure(config)
            
            # 设置较长的初始化时间，提高稳定性
            self.picam2.start(show_preview=False)
            time.sleep(3)  # 增加等待时间
            
            # 测试摄像头是否工作正常
            try:
                test_frame = self.picam2.capture_array()
                if test_frame is None or test_frame.size == 0:
                    logging.error("摄像头初始化失败，未能获取图像")
                    self.picam2.stop()
                    self.picam2.close()
                    self.picam2 = None
                    return False
            except Exception as e:
                logging.error(f"摄像头测试失败: {e}")
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
                return False
                
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            
            logging.info("ArUco检测初始化成功")
            self.aruco_initialized = True
            return True
        except Exception as e:
            logging.error(f"ArUco检测初始化失败: {e}")
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except:
                    pass
                self.picam2 = None
            return False
    
    def _run_aruco_detection(self):
        """运行ArUco标记检测"""
        if not self.aruco_initialized:
            logging.error("ArUco检测未初始化")
            return
        
        try:
            logging.info("开始检测ArUco标记...")
            
            # 进行10次标记检测尝试
            for _ in range(10):
                # 捕获图像
                frame = self.picam2.capture_array()
                
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # 检测标记
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                
                if ids is not None and len(ids) >= 2:
                    # 估计标记的姿态
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                    
                    # 计算两个标记之间的距离
                    if len(tvecs) >= 2:
                        # 获取两个标记的位置向量
                        pos1 = tvecs[0][0]
                        pos2 = tvecs[1][0]
                        
                        # 计算欧几里得距离
                        distance = np.linalg.norm(pos1 - pos2)
                        
                        logging.info(f"检测到 {len(ids)} 个ArUco标记:")
                        for i, marker_id in enumerate(ids):
                            logging.info(f"  - 标记 ID: {marker_id[0]}, 位置: {tvecs[i][0]}")
                        logging.info(f"  标记之间的距离: {distance:.3f} 米")
                        
                        # 设置标记已检测标志
                        self.aruco_markers_detected = True
                        # 检测成功后切换回RFID模式
                        self.current_mode = "RFID"
                        logging.info("ArUco标记检测成功，从视觉模式返回RFID模式")
                        break
                else:
                    logging.info("未检测到足够的ArUco标记（需要至少2个标记）")
                
                time.sleep(0.5)  # 降低检测频率
                
            if not self.aruco_markers_detected:
                logging.warning("未能成功检测到足够的ArUco标记，继续视觉模式")
                
        except Exception as e:
            logging.error(f"ArUco检测错误: {e}")
        finally:
            # 如果摄像头已初始化但未成功检测到标记，保持视觉模式
            pass
    
    def _check_vision_mode_timeout(self):
        """检查是否需要从视觉模式返回RFID模式"""
        # 只有在以下情况才会超时返回RFID模式：
        # 1. 当前是视觉模式
        # 2. 超过了设置的超时时间
        # 3. 未成功检测到标记
        if (self.current_mode == "VISION" and 
            time.time() - self.last_switch_time > self.vision_mode_duration and
            not self.aruco_markers_detected):
            self.current_mode = "RFID"
            logging.info(f"视觉模式超时，返回RFID模式")
            
            # 关闭摄像头
            if self.picam2 is not None:
                self.picam2.stop()
                self.aruco_initialized = False
    
    def get_tag_statistics(self, tag_id: str) -> Optional[dict]:
        """获取标签的统计信息"""
        if tag_id not in self.tag_history:
            return None
            
        history = self.tag_history[tag_id]
        if not history:
            return None
            
        rssi_values = [d.rssi_dbm for d in history]
        distances = [d.distance for d in history]
        
        return {
            "tag_id": tag_id,
            "rssi_mean": np.mean(rssi_values),
            "rssi_std": np.std(rssi_values),
            "distance_mean": np.mean(distances),
            "distance_std": np.std(distances),
            "sample_count": len(history),
            "last_update": datetime.fromtimestamp(history[-1].timestamp/1000).strftime('%Y-%m-%d %H:%M:%S') if history[-1].timestamp > 0 else "未知",
            "is_reliable": len(self.reliability_reasons[tag_id]) == 0 or self.reliability_reasons[tag_id][0] == "数据可靠",
            "reliability_status": "可靠" if (len(self.reliability_reasons[tag_id]) == 0 or self.reliability_reasons[tag_id][0] == "数据可靠") else "不可靠",
            "reasons": self.reliability_reasons[tag_id] if tag_id in self.reliability_reasons else []
        }

def extract_json(line):
    """尝试从不完整的行中提取有效的JSON"""
    try:
        # 尝试直接解析
        return json.loads(line)
    except json.JSONDecodeError:
        # 尝试找到JSON对象的起止位置
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, line)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
    return None

# 使用示例
def main():
    analyzer = RFIDAnalyzer()
    try:
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        logging.info("成功连接到串口 /dev/ttyAMA0")
    except Exception as e:
        logging.error(f"串口连接失败: {e}")
        return
        
    print("RFID分析器已启动，按Ctrl+C停止")
    
    # 模拟接收数据
    while True:
        try:
            # 从串口读取数据
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if not line:
                    time.sleep(0.01)
                    continue
                
                # 尝试解析JSON数据
                data = extract_json(line)
                if not data:
                    logging.error(f"无效JSON数据: {line}")
                    continue
                
                # 检查必要字段
                required_fields = ['id', 'rssi', 'rssiDbm', 'distance', 'isStable', 'timestamp']
                if not all(field in data for field in required_fields):
                    missing = [f for f in required_fields if f not in data]
                    logging.error(f"数据缺少必要字段: {missing}")
                    continue
                
                # 分析数据
                is_reliable = analyzer.add_data(data)
                reliability_status = "可靠" if is_reliable else "不可靠"
                
                # 获取统计信息
                stats = analyzer.get_tag_statistics(data['id'])
                if stats:
                    print("\n标签统计信息:")
                    print(f"标签ID: {stats['tag_id']}")
                    print(f"RSSI均值: {stats['rssi_mean']:.1f}dBm")
                    print(f"RSSI标准差: {stats['rssi_std']:.2f}")
                    print(f"距离均值: {stats['distance_mean']:.2f}米")
                    print(f"距离标准差: {stats['distance_std']:.4f}米")
                    print(f"样本数: {stats['sample_count']}")
                    print(f"最后更新: {stats['last_update']}")
                    print(f"可靠性状态: {stats['reliability_status']}")
                    print(f"原因: {', '.join(stats['reasons'])}")
                
                # 打印当前模式
                print(f"当前定位模式: {analyzer.current_mode}")
                
            else:
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n程序已停止")
            if analyzer.picam2 is not None:
                analyzer.picam2.stop()
            break
        except Exception as e:
            logging.error(f"错误: {e}")
            continue

if __name__ == "__main__":
    main()
