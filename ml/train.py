from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import dvc.api
from dvclive import Live

from pathlib import Path
import random
from tqdm import tqdm

from datasets.demo import CatsAnDogs
from models.resnets import CustomResnet

params = dvc.api.params_show()

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    model.to(device)

    with Live() as live:
        live.log_param('num_epochs', num_epochs)
        live.log_param('learning_rate', LEARNING_RATE)
        live.log_param('trasformations', 'Normalize')
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 20)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = torch.argmax(labels, 1)
                    
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.long() / len(dataloaders[phase].dataset)
                
                live.log_metric(f'{phase}_loss', epoch_loss)
                live.log_metric(f'{phase}_acc', epoch_acc.cpu().item())
                live.next_step()

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        print("Training complete")
    return model


if __name__ == '__main__':
    LEARNING_RATE = params['train']['LEARNING_RATE']
    NUM_EPOCHS = params['train']['NUM_EPOCHS']
    
    TRAIN_IMAGES_PATH = Path('data/interim/train')
    train_images_paths = list(TRAIN_IMAGES_PATH.iterdir())
    random.shuffle(train_images_paths)
    
    TEST_IMAGES_PATH = Path('data/interim/test')
    test_images_paths = list(TEST_IMAGES_PATH.iterdir())
    random.shuffle(test_images_paths)
    
    transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_set = CatsAnDogs(train_images_paths[:20000], transforms=ToTensor())
    test_set = CatsAnDogs(test_images_paths[:5000], transforms=ToTensor())
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    
    model = CustomResnet(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model = train_model(
        model,
        dataloaders,
        num_epochs=NUM_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        device='mps'
    )