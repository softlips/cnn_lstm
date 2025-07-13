import time
import datetime
import platform
import struct
import sensor.lib.device_model as deviceModel
from sensor.lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
from sensor.lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver

def onUpdate(devicemodel):
    pass

def ini_device():
    """
    初始化一个设备模型   Initialize a device model
    """
    device = deviceModel.DeviceModel(
        "我的JY901",
        WitProtocolResolver(),
        JY901SDataProcessor(),
        "51_0"
    )

    if (platform.system().lower() == 'darwin'):
        # ls /dev/tty.*
        device.serialConfig.portName = "/dev/tty.usbserial-14130"  # 设置串口   Set serial port
    else:
        device.serialConfig.portName = "COM10"  # 设置串口   Set serial port

    device.serialConfig.baud = 115200  # 设置波特率  Set baud rate
    device.openDevice()  # 打开串口   Open serial port
    device.AccelerationCalibration()  # Acceleration calibration
    print("加计校准结束")
    device.writeReg(0x52)     #设置z轴角度归零
    device.writeReg(0x65)  # 设置安装方向:水平
    # device.writeReg(0x66)  # 设置安装方向:垂直

    device.dataProcessor.onVarChanged.append(onUpdate)  # 数据更新事件 Data update event

    return device