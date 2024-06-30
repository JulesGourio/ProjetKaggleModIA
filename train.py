import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from torch.utils.data import Subset, DataLoader

def train(num_epochs, train_loader, test_loader, model, criterion, optimizer, scheduler, device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()

    best_val_accuracy = 0.0  # Initialize best validation accuracy

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            if images is None or labels is None:
                continue

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                progress_bar.set_postfix({'Loss': loss.item()})
                progress_bar.update()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                if images is None or labels is None:
                    continue

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)


        if scheduler is not None:
            scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

        # Check if this is the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = f'{model.name}_best_{num_epochs}epochs_{optimizer.__class__.__name__}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved: {best_model_path} with validation accuracy: {best_val_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time

    print(f'Total training time: {total_time:.2f} seconds')

    # Save the final model
    final_model_path = f'{model.name}_final_{num_epochs}epochs_{optimizer.__class__.__name__}_time{total_time:.2f}seconds.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved: {final_model_path}')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


    return train_losses, val_losses, train_accuracies, val_accuracies




def trainKF(kf, dataset, model, criterion, optimizer, scheduler, device, batch_size):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()

    best_val_accuracy = 0.0  # Initialize best validation accuracy

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        num_folds = kf.n_splits

        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {fold+1}/{num_folds}")
        for i, (images, labels) in enumerate(progress_bar):
            if images is None or labels is None:
                continue
                #Afficher la taille des images



            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                progress_bar.set_postfix({'Loss': loss.item()})
                progress_bar.update()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                if images is None or labels is None:
                    continue

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)


        if scheduler is not None:
            scheduler.step(avg_val_loss)

        print(f'Epoch [{fold + 1}/{num_folds}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

        # Check if this is the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            #best_model_path = f'{model}_best_{num_folds}epochs_{optimizer.__class__.__name__}.pth'
            best_model_path = f'pretrained_best_{num_folds}epochs_{optimizer.__class__.__name__}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved: {best_model_path} with validation accuracy: {best_val_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time

    print(f'Total training time: {total_time:.2f} seconds')

    # Save the final model
    final_model_path = f'pretrained_final_{num_folds}epochs_{optimizer.__class__.__name__}_time{total_time:.2f}seconds.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved: {final_model_path}')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_folds + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_folds + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_folds + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_folds + 1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
