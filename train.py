import torch
from torch import nn, optim
from tqdm import tqdm
import logging
from config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
from gesture_dataset import get_dataloaders
from model import GestureRecognitionModel

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播（获取主输出和辅助输出）
        logits, aux_logits = model(frames)
        
        # 计算主损失和辅助损失
        main_loss = criterion(logits, labels)
        aux_loss = criterion(aux_logits, labels)
        loss = main_loss + 0.3 * aux_loss  # 辅助损失权重0.3
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 计算准确率
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total:.4f}"
        })
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            # 前向传播（验证时只有主输出）
            logits = model(frames)
            loss = criterion(logits, labels)
            
            # 计算准确率
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader), correct / total

def main():
    """主训练函数"""
    try:
        # 初始化模型
        model = GestureRecognitionModel().to(DEVICE)
        
        # 首先冻结CNN部分
        model.freeze_cnn()
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5, verbose=True
        )
        
        # 获取数据加载器
        train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
        logging.info(f"数据加载完成，训练批次大小: {BATCH_SIZE}")
        
        # 训练循环
        best_val_acc = 0.0
        no_improve_epochs = 0
        
        for epoch in range(NUM_EPOCHS):
            # 如果到第10个epoch，解冻CNN
            if epoch == 10:
                logging.info("解冻CNN层...")
                model.unfreeze_cnn()
                # 更新优化器以包含所有参数
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=LEARNING_RATE * 0.1,  # 降低学习率
                    weight_decay=0.01
                )
            
            # 训练一个epoch
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE, epoch
            )
            
            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 打印信息
            logging.info(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, "best_model.pth")
                logging.info(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            # 早停
            if no_improve_epochs >= 10:
                logging.info("10个epoch没有改善，停止训练")
                break
    
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()