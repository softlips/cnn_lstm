import torch

# 数据集路径和类别（需与实际文件夹名匹配，注意下划线/短横线）
DATA_DIR = "data/rgb_blur"
CLASS_NAMES = ["scroll_down", "scroll_up", "scroll_left", "scroll_right", "zoom_in", "zoom_out"]  # 修正为实际文件夹名（假设你的数据是下划线，如scroll_down）

print(f"Data directory: {DATA_DIR}")  # 运行脚本时检查输出是否正确
# 超参数
BATCH_SIZE = 8      # 批量大小
SEQUENCE_LENGTH = 5 # 修改：每手势取5帧（原为10，为解决超时问题暂时减小）
NUM_EPOCHS = 10     # 训练轮数
LEARNING_RATE = 0.001 # 学习率
LSTM_HIDDEN = 512   # LSTM隐藏层大小
DROPOUT = 0.5       # 丢弃率
NUM_CLASSES = len(CLASS_NAMES)

# 设备（自动检测CUDA）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
