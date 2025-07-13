import sys
import cv2
import os
from datetime import datetime
import time
import numpy as np
from multiprocessing import Process, Queue, Barrier
import threading
from PIL import Image
import sensor_api
import pandas as pd
import emg_api

current_datetime = datetime.now()
year = current_datetime.year
mouth = current_datetime.month
day = current_datetime.day

"""#camera setting
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

# 获取视频帧的宽度和高度
frame_width1 = int(cam1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height1 = int(cam1.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width2 = int(cam2.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height2 = int(cam2.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)"""

subject = "subject11"
scene = "scene9"


def get_max_number():
    folder_ = f"../Dataset/{subject}/{scene}"
    os.makedirs(folder_, exist_ok=True)
    folder_count = sum(
        1
        for entry in os.listdir(folder_)
        if os.path.isdir(os.path.join(folder_, entry))
    )
    folder_count = folder_count + 1
    return folder_count


rgb_id = get_max_number()

save_folder_root_up = f"../Dataset/{subject}/{scene}/{rgb_id}/up_hand"
save_folder_root_down = f"../Dataset/{subject}/{scene}/{rgb_id}/down_hand"
# save_folder_root_refer = f"../Dataset/{subject}/{scene}/{rgb_id}/refer"
save_folder_root_sensor = f"../Dataset/{subject}/{scene}/{rgb_id}/sensor"
save_folder_root_emg = f"../Dataset/{subject}/{scene}/{rgb_id}/emg"
folder_list = [
    save_folder_root_up,
    save_folder_root_down,
    # save_folder_root_refer,
    save_folder_root_sensor,
    save_folder_root_emg,
]

#同步（2个摄像头+1个传感器——1个emg）
barrier_c = Barrier(2)
barrier_a = Barrier(4)

frame1_queue = Queue()
frame2_queue = Queue()
# frame3_queue = Queue()
sensor_queue = Queue()
emg_queue=Queue()
signal_queue = Queue()


def process_camera(
    camera_index, window_name, frame_queue, signal_queue, barrier_c, barrier_a
):
    # 打开指定的摄像头
    cap = cv2.VideoCapture(camera_index)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if 24 <= fps <= 26 :
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    synchronized_fps = 12.0

    # 计算帧之间的等待时间
    frame_duration = 1.0 / synchronized_fps

    # 初始化帧计数和时间
    frame_count = 0
    fps = 0
    fps_start_time = time.time()

    i = 0
    drop_frame = 30
    ret, frame = cap.read()
    if ret:
        cv2.imshow(window_name, frame)
    while True:
        start_time = time.time()
        # 逐帧读取摄像头的视频
        ret, frame = cap.read()

        if ret:
            # 帧计数增加
            frame_count += 1
            current_time = time.time()
            fps_elapsed_time = current_time - fps_start_time

            # 每过 1 秒计算一次 FPS
            """if fps_elapsed_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_start_time = current_time"""

            # 在帧上显示 FPS
            # cv2.putText(frame, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 计算处理一帧的时间
            elapsed_time = time.time() - start_time

            if camera_index == 0:
                cv2.moveWindow(window_name, 0, 0)
            if camera_index == 1:
                cv2.moveWindow(window_name, 640, 0)
            cv2.imshow(window_name, frame)

            if i == drop_frame + 1:
                barrier_a.wait()

            if i > drop_frame and i <= drop_frame + 121:
                barrier_c.wait()
                # 如果处理时间小于帧时间间隔，则等待剩余时间
                if elapsed_time < frame_duration:
                    time.sleep(frame_duration - elapsed_time)
                else:
                    print("Warning:Camera frame is not sync with set fps!")

                print(f"开始录制！--{camera_index}--{i}")
                if i > (drop_frame + 1):
                    frame_queue.put(frame)

            i += 1

            if i > (drop_frame + 120 + 1):
                signal_queue.put("x")
                print(f"结束录制{camera_index}")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def process_sensor():
    device = sensor_api.ini_device()
    synchronized_fps = 100.0
    frame_duration = 1.0 / synchronized_fps

    barrier_a.wait()


    start_time = time.time()
    print("传感器开始")
    while True:
        if not signal_queue.empty():
            print("传感器接收数据结束！")
            break
        data = device.deviceData
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_duration:
            pass
        else:
            sensor_queue.put(data)
            start_time = time.time()



def process_emg():
    device = emg_api.ini_emg_device()
    barrier_a.wait()

    print("EMG数据采集开始")

    while True:
        if not signal_queue.empty():
            print("EMG采集结束")
            break

        # 获取所有可用EMG数据
        while True:
            emg_data = emg_api.get_emg_data(device)
            if emg_data is None:
                break
            # emg_data是包含8个值的列表
            emg_queue.put(emg_data)

    emg_api.close_emg_device(device)


def save_data_from_queue(queue1, queue2):
    i = 0
    sensor_list = []
    emg_list=[]

    while True:
        if not queue1.empty() and not queue2.empty() :
            frame1 = queue1.get()
            frame2 = queue2.get()
            # frame3 = queue3.get()

            image1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            image1.save(f"{save_folder_root_up}/{i}.jpg")
            image2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            image2.save(f"{save_folder_root_down}/{i}.jpg")
            # image3 = Image.fromarray(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB))
            # image3.save(f"{save_folder_root_refer}/{i}.jpg")
            '''cv2.imwrite(f"{save_folder_root_up}/{i}.jpg", frame1)
            cv2.imwrite(f"{save_folder_root_down}/{i}.jpg", frame2)
            cv2.imwrite(f"{save_folder_root_refer}/{i}.jpg", frame3)'''
            i += 1

        if not sensor_queue.empty():
            sensor = sensor_queue.get()
            sensor_list.append(sensor)
        # 保存EMG数据
        if not emg_queue.empty():
            emg = emg_queue.get()
            emg_list.append(emg)


        if (
            queue1.empty()
            and queue2.empty()
            # and queue3.empty()
            and sensor_queue.empty()
            and emg_queue.empty()
            and not signal_queue.empty()
        ):
            break
    # 保存传感器数据
    if sensor_list:
        sensor_df = pd.DataFrame(sensor_list)
        sensor_df.to_csv(f"{save_folder_root_sensor}/sensor.csv", index=True, encoding="utf-8")
        print(f"传感器数据已保存至 {save_folder_root_sensor}/sensor.csv")

    # 保存EMG数据
    if emg_list:
        emg_df = pd.DataFrame(emg_list, columns=[f"emg{i+1}" for i in range(8)])
        emg_df.to_csv(f"{save_folder_root_emg}/emg.csv", index=True, encoding="utf-8")
        print(f"EMG数据已保存至 {save_folder_root_emg}/emg.csv")

    # df = pd.DataFrame(sensor_list)
    # df.to_csv(f"{save_folder_root_sensor}/sensor.csv", index=True, encoding="utf-8")
    # emg_df = pd.DataFrame(emg_list, columns=[f"emg{i + 1}" for i in range(8)])
    # emg_df.to_csv(f"{save_folder_root_emg}/emg.csv", index=True, encoding="utf-8")
if __name__ == "__main__":

    for folder in folder_list:
        os.makedirs(folder, exist_ok=True)

    process1 = Process(
        target=process_camera,
        args=(0, "Camera 1", frame1_queue, signal_queue, barrier_c, barrier_a),
    )
    process2 = Process(
        target=process_camera,
        args=(2, "Camera 2", frame2_queue, signal_queue, barrier_c, barrier_a),
    )
    # process3 = Process(
    #     target=process_camera,
    #     args=(2, "Camera 3", frame3_queue, signal_queue, barrier_c, barrier_a),
    # )
    thread1 = threading.Thread(target=process_sensor)
    thread_emg=threading.Thread(target=process_emg)
    thread2 = threading.Thread(
        target=save_data_from_queue,
        args=(
            frame1_queue,
            frame2_queue,
            # frame3_queue,
        ),
    )


    process1.daemon = True
    process2.daemon = True
    # process3.daemon = True
    thread1.daemon = True
    thread_emg.daemon=True
    thread2.daemon = True

    thread2.start()
    thread1.start()#传感器
    thread_emg.start()
    process1.start()
    process2.start()
    # process3.start()


    process1.join()
    process2.join()
    # process3.join()
    thread1.join()
    thread_emg.join()
    thread2.join()

    print("所有数据采集完成")
