import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, LSTM_HIDDEN, DROPOUT

class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练Inception-v3
        self.cnn = models.inception_v3(pretrained=True, aux_logits=True)
        self.cnn.fc = nn.Identity()  # 移除最后的分类层，保留2048维特征
        self.cnn.AuxLogits.fc = nn.Linear(768, NUM_CLASSES)  # 修改辅助分类器输出维度
        
        # 时序特征提取
        self.lstm = nn.LSTM(
            input_size=2048,  # Inception-v3输出维度
            hidden_size=LSTM_HIDDEN,
            num_layers=2,     # 使用2层LSTM
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, LSTM_HIDDEN),  # 2是因为双向LSTM
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN // 2, NUM_CLASSES)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # 1. CNN特征提取
        # 重塑输入以便于CNN处理
        x = x.view(batch_size * seq_len, c, h, w)
        
        # CNN前向传播
        if self.training:
            cnn_features, aux_logits = self.cnn(x)
            # 处理辅助输出，只使用序列中间帧的预测
            aux_logits = aux_logits.view(batch_size, seq_len, -1)
            aux_logits = aux_logits[:, seq_len//2, :]  # 取中间帧的预测
        else:
            cnn_features = self.cnn(x)
        
        # 2. 重组时序数据
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # 3. LSTM处理时序特征
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # 4. 获取最后一个时间步的隐藏状态
        # 对于双向LSTM，需要连接前向和后向的最后隐藏状态
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # 5. 特征融合
        fused_features = self.fusion(last_hidden)
        
        # 6. 分类
        logits = self.classifier(fused_features)
        
        if self.training:
            return logits, aux_logits
        return logits

    def freeze_cnn(self):
        """冻结CNN参数"""
        for param in self.cnn.parameters():
            param.requires_grad = False
            
    def unfreeze_cnn(self):
        """解冻CNN参数"""
        for param in self.cnn.parameters():
            param.requires_grad = True