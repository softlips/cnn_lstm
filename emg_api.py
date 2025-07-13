import serial
import platform
import time
import struct

def ini_emg_device():
    """初始化EMG设备（指定串口6，兼容所有pyserial版本）"""
    # 根据系统设置串口（Windows: COM6，macOS: /dev/tty.usbserial-6）
    if platform.system().lower() == 'darwin':
        port = "/dev/tty.usbserial-6"
    else:
        port = "COM6"

    try:
        # 使用serial模块定义的常量（最兼容的方式）
        # 先尝试直接引用属性（高版本支持）
        try:
            parity = serial.PARITY_NONE
            stopbits = serial.STOPBITS_1
            bytesize = serial.EIGHTBITS
        except AttributeError:
            # 低版本兼容：用字符串替代
            parity = 'N'  # 无校验
            stopbits = 1  # 1位停止位
            bytesize = 8  # 8位数据位

        # 初始化串口
        device = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=0.1,
            parity=parity,
            stopbits=stopbits,
            bytesize=bytesize
        )

        if not device.is_open:
            device.open()

        # 设备初始化（根据实际协议调整）
        time.sleep(0.5)  # 等待设备响应
        device.write(b'\xAA\x55\x01\x00\x00')  # 启动采集指令（示例）
        time.sleep(0.1)

        print(f"EMG设备已连接至 {port}")
        return device

    except Exception as e:
        print(f"EMG初始化失败: {e}")
        raise

def get_emg_data(device):
    """读取8通道EMG数据"""
    if not device or not device.is_open:
        raise ValueError("EMG设备未连接")

    try:
        # 假设数据格式：1字节头 + 8×2字节数据 + 1字节校验（共17字节）
        raw_data = device.read(17)
        if len(raw_data) == 17 and raw_data[0] == 0xAA:  # 校验头标识
            emg_values = []
            for i in range(8):
                # 解析16位有符号整数（小端模式）
                val = struct.unpack('<h', raw_data[1 + i*2 : 3 + i*2])[0]
                emg_values.append(val)
            return emg_values
        else:
            device.flushInput()  # 清空无效数据
            return None

    except Exception as e:
        print(f"EMG数据读取错误: {e}")
        return None

def close_emg_device(device):
    """关闭EMG设备连接"""
    if device and device.is_open:
        device.close()
        print("EMG设备已关闭")