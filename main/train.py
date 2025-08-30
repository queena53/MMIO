import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import json
import math
from densenet import DenseNet3D
from model_medicalNet import resnet10, resnet18, resnet101
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, resnet
from unet import UNet3DClassifier
from data import get_folds
from datetime import datetime  # 导入 datetime 模块
from sklearn.metrics import confusion_matrix

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


# ----------------------------
# learning rate
# ----------------------------


class CosineAnnealingLRTMult(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, T_mult=1, last_epoch=-1):
        self.T_max = T_max          # 初始周期长度
        self.eta_min = eta_min      # 最小学习率
        self.T_mult = T_mult        # 周期倍增系数
        self.current_cycle = 0      # 当前周期计数
        self.T_i = T_max            # 当前周期的长度
        super(CosineAnnealingLRTMult, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch >= self.T_i:
            # 每个周期结束时重置周期长度，并更新为 T_mult 倍
            self.current_cycle += 1
            self.last_epoch = 0
            self.T_i = self.T_max * (self.T_mult ** self.current_cycle)

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_i)) / 2
                for base_lr in self.base_lrs]

    
class CosineAnnealingLRTMultWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, T_mult=1, decay_rate=0.9, last_epoch=-1):
        self.T_max = T_max          # 初始周期长度
        self.eta_min = eta_min      # 最小学习率
        self.T_mult = T_mult        # 周期倍增系数
        self.decay_rate = decay_rate  # 衰减率
        self.current_cycle = 0      # 当前周期计数
        self.T_i = T_max            # 当前周期的长度
        super(CosineAnnealingLRTMultWithDecay, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch >= self.T_i:
            # 每个周期结束时，重置周期长度、衰减初始学习率，并更新为 T_mult 倍
            self.current_cycle += 1
            self.last_epoch = 0
            self.T_i = self.T_max * (self.T_mult ** self.current_cycle)
            self.base_lrs = [base_lr * self.decay_rate for base_lr in self.base_lrs]

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_i)) / 2
                for base_lr in self.base_lrs]
    

#cosine_annealing_warmup
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_epochs:
                # 線性 warmup
                lr = base_lr * (self.last_epoch + 1) / self.warmup_epochs
            else:
                # 餘弦衰減
                progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine_decay
            lrs.append(lr)
        return lrs

class OneCycleLRScheduler(_LRScheduler):
    """
    PyTorch-style One Cycle Learning Rate Scheduler.
    Implements one cycle policy where the learning rate increases from initial to max_lr, 
    then decreases to min_lr using cosine annealing.
    
    Args:
        optimizer: torch optimizer.
        max_lr: Peak learning rate.
        total_epochs: Total number of training epochs.
        pct_start: Percentage of total_epochs for increasing phase. Default: 0.3
        min_lr: Final minimum learning rate. Default: 1e-6
        last_epoch: The index of last epoch. Default: -1
    """
    def __init__(self, optimizer, max_lr, total_epochs, pct_start=0.0, min_lr=1e-10, last_epoch=-1):
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.min_lr = min_lr
        self.peak_epoch = int(total_epochs * pct_start)
        self.final_phase_epochs = total_epochs - self.peak_epoch
        super(OneCycleLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1  # current_epoch starts from 0
        lrs = []
        for base_lr in self.base_lrs:
            if current_epoch <= self.peak_epoch:
                # Linear warm-up phase
                lr = base_lr + (self.max_lr - base_lr) * current_epoch / self.peak_epoch
            else:
                # Cosine annealing from max_lr to min_lr
                t = current_epoch - self.peak_epoch
                cosine_decay = 0.5 * (1 + math.cos(math.pi * t / self.final_phase_epochs))
                lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            lrs.append(lr)
        return lrs

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

# ----------------------------
# Training function
# ----------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (x, y, _) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

def validate(model, dataloader, criterion, device, fold_idx, use_wandb=False, class_names=None, epoch=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    best_loss = float('inf')
    current_results = []  # 👉 當前 epoch 的所有結果
    best_epoch_results = []  # 👉 所有最佳 epoch 的結果
    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, (x, y, filenames) in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            probs = torch.softmax(out, dim=1)
            preds = out.argmax(1)

            # 累加結果與 loss
            total_loss += loss.item() * x.size(0)
            epoch_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

            for filename, pred, true, prob in zip(filenames, preds.cpu().numpy(), y.cpu().numpy(), probs.cpu().numpy()):
                current_results.append({
                    "Filename": filename,
                    "Predicted": int(pred),
                    "True": int(true),
                    "Probabilities": prob.tolist()
                })

    avg_loss = total_loss / total
    avg_acc = correct / total

    # 🔥 如果這是目前最佳 loss，保存整個 val 結果
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch_results = current_results.copy()

        # 保存為 txt 或 csv
        project_name = config["project_name"]
        wandb_name = wandb.run.name if config["use_wandb"] else "default"
        output_dir = f"output/{project_name}/{wandb_name[:-7]}"
        os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在
        output_file = os.path.join(output_dir, f"fold_{fold_idx}_best_val_epoch.txt")  # 加上 fold_idx

        with open(output_file, "w") as f:
            f.write(f"[Epoch {epoch}] Best Validation Results (Total: {len(best_epoch_results)}):\n")
            for r in best_epoch_results:
                f.write(f"{os.path.join(*r['Filename'].split(os.sep)[-3:])}: Predicted={r['Predicted']}, True={r['True']}, Probabilities={r['Probabilities']}\n")
        print(f"\n✅ Best epoch {epoch} results saved to {output_file}")

        # 上傳 wandb
        if use_wandb:
            table_data = [[r["Filename"], r["Predicted"], r["True"], r["Probabilities"]] for r in best_epoch_results]
            wandb.log({
                "best_val_table": wandb.Table(data=table_data, columns=["Filename", "Predicted", "True", "Probabilities"]),
                "best_val_loss": best_loss
            }, step=epoch)

    return avg_loss, avg_acc

# ----------------------------
# Training Loop (Per Fold)
# ----------------------------

def train_all_folds(config):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    folds = get_folds(config)

    for fold_idx, fold in enumerate(folds):

        print(f"\n Training Fold {fold_idx}")

        if config["use_wandb"]:
            wandb.init(project=config["project_name"], name=f"desnet201_newfold_{fold_idx}", config=config)
            #  wandb.init(project=config["project_name"], name=f"test_{fold_idx}", config=config)

        # 创建 wandb 表格
        fold_table = wandb.Table(columns=["Fold", "Type", "Patient", "Diagnosis", "Instrument"])

        # Train 資訊
        for patient, info in fold["train_info"].items():
            
            fold_table.add_data(fold_idx + 1, "Train", patient, info["diagnosis"], info["instrument"])

        # Val 資訊
        for patient, info in fold["val_info"].items():
            fold_table.add_data(fold_idx + 1, "Val", patient, info["diagnosis"], info["instrument"])

        # 紀錄到 wandb

        wandb.log({"fold_table": fold_table})


        # model = resnet18(  
        #     sample_input_D=128,
        #     sample_input_H=128,
        #     sample_input_W=128,
        #     shortcut_type='B',
        #     no_cuda=False,
        #     num_seg_classes=config["num_classes"]
        #     in_channels=120
        # ).to(device)

      

        # # Fix: 正確使用 kaiming 初始化
        # for m in model.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        model = DenseNet201(
        spatial_dims=3,
        in_channels=1,  
        out_channels=2,  # 二分類
        init_features=32,
        growth_rate=16,
        block_config=(2, 2, 2, 2),  # 可以適當調整為 (2,2,2,1) 或 (2,2,1)
        dropout_prob=0.5  # dropout 機率
        ).to(device)

        # model = DenseNet3D(
        #     in_channels=1,
        #     num_classes=config["num_classes"],
        #     growth_rate=32,
        #     block_config=(6, 12, 24, 16),
        #     num_init_features=64,
        #     bn_size=4,
        #     drop_rate=0.0,
        # ).to(device)

        # model = resnet.resnet101(
        # spatial_dims=3,          # 使用 3D 卷積
        # n_input_channels=1,    # 輸入通道數為 120
        # num_classes=2            # 二分類
        # ).to(device)

        # model = UNet3DClassifier(
        # in_channels=1,
        # num_classes=len(config["selected_classes"]),
        # base_channels=32  # 可調整為 16 或 64
        # ).to(device)

        train_loader = DataLoader(fold["train_dataset"], batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(fold["val_dataset"], batch_size=config["batch_size"], shuffle=False)

        # optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)

        # criterion = nn.CrossEntropyLoss(label_smoothing =0.1)  # 使用 label smoothing
        criterion = FocalLoss(alpha=1, gamma=2)

        optimizer = torch.optim.Adam(model.parameters(),
                             lr=config["learning_rate"],
                             weight_decay=config["l2_lambda"]) 
        
        # optimizer = torch.optim.AdamW(
        # model.parameters(), 
        # lr=config["learning_rate"], 
        # weight_decay=config.get('weight_decay', 1e-4)
        # )
        
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=10, cycle_mult=2, max_lr=0.001, min_lr=0, warmup_steps=9, gamma=0.9)

        # scheduler=CosineAnnealingLRTMultWithDecay(optimizer, T_max=5, eta_min=0, T_mult=2, decay_rate=0.9, last_epoch=-1)

        # scheduler = CosineAnnealingLRTMult(optimizer, eta_min=0,T_mult=2, T_max=30, last_epoch=-1)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=10, last_epoch=-1)
        
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-4)


        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3,steps_per_epoch=170, epochs=50,pct_start=0.3,anneal_strategy='cos',div_factor=25.0,final_div_factor=10000.0)
       
        # scheduler = CosineWarmupScheduler(optimizer=optimizer,warmup_epochs=10,max_epochs=50,min_lr=1e-6)

        scheduler = OneCycleLRScheduler(optimizer, max_lr=4e-5, total_epochs=config["epochs"], pct_start=0.15, min_lr=1e-10)

        
        best_loss = float('inf')  # 初始化最小验证损失
        patience = config.get("early_stop_patience", 100)  # 从配置中读取 patience，默认为 10
        patience_counter = 0  # 记录验证集性能未提升的次数

        best_acc = 0

        l1_lambda = config["l1_lambda"]  # 從 config 中讀取 l1_lambda

        for epoch in range(config["epochs"]):
            print(f"\n[Epoch {epoch + 1}]")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, 
                fold_idx=fold_idx,  # 傳遞 fold_idx
                use_wandb=config["use_wandb"], 
                class_names=config["selected_classes"], 
                epoch=epoch + 1
            )
            
            scheduler.step()

            print(f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

            if config["use_wandb"]:
                wandb.log({
                    "epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })

            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0  # 重置 patience 计数器
                # 保存模型
                wandb_name = wandb.run.name if config["use_wandb"] else "default"
                project_name = config["project_name"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                output_path = f"output/{project_name}/{wandb_name[:-7]}/{wandb_name}_model.pth"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save(model.state_dict(), output_path)
                print(f"Model saved at epoch {epoch + 1} with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Early stopping patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if config["use_wandb"]:
            wandb.finish()


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    train_all_folds(config)
