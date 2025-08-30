### val.py
import torch
import torch.nn as nn
import json
from model import ResNet101
from data import get_dataloaders


def run_validation():
    with open("config.json") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = 0
    _, val_loader = get_dataloaders(config, fold)

    model = ResNet101(num_classes=config["num_classes"])
    model.load_state_dict(torch.load(f"{config['output_dir']}/model.pth"))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    run_validation()