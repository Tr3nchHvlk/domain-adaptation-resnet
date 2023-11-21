import torch 
from torch import optim
from torch.optim import Optimizer
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from torchvision import transforms
from torchvision.models import ResNet
from torchvision.models import resnet34
from torchvision.datasets import ImageFolder

from numpy import log10, floor

from PIL import Image

def pad_to_square(stock):
    x, y = stock.size
    d = max(x, y)
    padded = Image.new(mode='RGB', size=(d, d), color=(255, 255, 255))
    padded.paste(stock, ((d - x) // 2, (d - y) // 2))
    return padded

image_refactor = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

real_test_set = ImageFolder(root='./data/real_test', transform=image_refactor)
real_train_set = ImageFolder(root='./data/real_train', transform=image_refactor)
sketch_test_set = ImageFolder(root='./data/sketch_test', transform=image_refactor)
sketch_train_set = ImageFolder(root='./data/sketch_train', transform=image_refactor)

real_test_loader = DataLoader(real_test_set, batch_size=64, shuffle=True)
real_train_loader = DataLoader(real_train_set, batch_size=64, shuffle=True)
sketch_test_loader = DataLoader(sketch_test_set, batch_size=64, shuffle=True)
sketch_train_loader = DataLoader(sketch_train_set, batch_size=64, shuffle=True)

def train(
    model: ResNet, 
    src_loader: DataLoader, 
    optimizer: Optimizer,
    loss_fn = CrossEntropyLoss(),
    epochs: int = 1,
    early_stop_delta: float = -1,
    checkpoint_dir_prefix: str = None,
):
    model.train()
    train_losses = []
    train_accs = []
    epochs_digits = int(floor(log10(epochs)) + 1)

    for e in range(epochs):
        r_loss = 0.0
        train_loss = 0.0
        correct_matches = 0
        y_dim = 0

        for i, (X, y) in enumerate(src_loader):
            optimizer.zero_grad()

            _y_probs = model.forward(X)
            loss = loss_fn(_y_probs, y)

            with torch.no_grad():
                r_loss += loss.item()
                train_loss += loss.item()
                _y = _y_probs.softmax(dim=1).argmax(dim=1)
                correct_matches += (y == _y).sum().item()
                y_dim += y.size(0)

            loss.backward()
            optimizer.step()

        if checkpoint_dir_prefix is not None:
            torch.save(model.state_dict(), "{}/{:0{}d}.pth".format(checkpoint_dir_prefix, e, epochs_digits))

        train_accs.append(correct_matches / y_dim)

        # early stopping
        if (len(train_losses) > 0) and ((train_losses[-1] - train_loss) < (train_losses[-1] * early_stop_delta)):
            train_losses.append(train_loss)

            with torch.no_grad():
                train_losses.append(0.0)
                correct_matches = 0
                y_dim = 0

                for i, (X, y) in enumerate(src_loader):
                    _y_probs = model.forward(X)
                    train_losses[-1] += loss_fn(_y_probs, y).item()
                    _y = _y_probs.softmax(dim=1).argmax(dim=1)
                    correct_matches += (y == _y).sum().item()
                    y_dim += y.size(0)

                train_accs.append(correct_matches / y_dim)
            
            return (model, train_losses[1:], train_accs[1:])

        train_losses.append(train_loss)

    with torch.no_grad():
        train_losses.append(0.0)
        correct_matches = 0
        y_dim = 0

        for i, (X, y) in enumerate(src_loader):
            _y_probs = model.forward(X)
            train_losses[-1] += loss_fn(_y_probs, y).item()
            _y = _y_probs.softmax(dim=1).argmax(dim=1)
            correct_matches += (y == _y).sum().item()
            y_dim += y.size(0)

        train_accs.append(correct_matches / y_dim)
    
    return (model, train_losses[1:], train_accs[1:])

def test(
    model: ResNet, 
    src_loader: DataLoader, 
    loss_fn = CrossEntropyLoss()
):
    model.eval()

    test_loss = 0.0
    test_acc = 0.0

    correct_matches = 0
    y_dim = 0

    for i, (X, y) in enumerate(src_loader):
        _y_probs = model.forward(X)
        test_loss += loss_fn(_y_probs, y).item()
        _y = _y_probs.softmax(dim=1).argmax(dim=1)
        correct_matches += (y == _y).sum().item()
        y_dim += y.size(0)

    test_acc = correct_matches / y_dim

    return (test_loss, test_acc)

def train_and_validate(
    model: ResNet, 
    src_loader: DataLoader, 
    val_loader: DataLoader, 
    optimizer: Optimizer,
    loss_fn = CrossEntropyLoss(),
    epochs: int = 1,
    early_stop_margin: float = -1,
    checkpoint_dir_prefix: str = None,
):
    model.train()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    # loss_deltas = []
    epochs_digits = int(floor(log10(epochs)) + 1)

    for e in range(epochs):
        r_loss = 0.0
        train_losses.append(0.0)
        correct_matches = 0
        y_dim = 0

        for i, (X, y) in enumerate(src_loader):
            optimizer.zero_grad()
            _y_probs = model.forward(X)
            loss = loss_fn(_y_probs, y)

            with torch.no_grad():
                r_loss += loss.item()
                train_losses[-1] += loss.item()
                _y = _y_probs.softmax(dim=1).argmax(dim=1)
                correct_matches += (y == _y).sum().item()
                y_dim += y.size(0)

            loss.backward()
            optimizer.step()

        if checkpoint_dir_prefix is not None:
            torch.save(model.state_dict(), "{}/{:0{}d}.pth".format(checkpoint_dir_prefix, e, epochs_digits))

        train_accs.append(correct_matches / y_dim)

        # validation
        with torch.no_grad():
            val_losses.append(0.0)
            correct_matches = 0
            y_dim = 0

            for i, (X, y) in enumerate(val_loader):
                _y_probs = model.forward(X)
                val_losses[-1] += loss_fn(_y_probs, y).item()
                _y = _y_probs.softmax(dim=1).argmax(dim=1)
                correct_matches += (y == _y).sum().item()
                y_dim += y.size(0)

            val_accs.append(correct_matches / y_dim)

        # early stopping alternative (on loss delta percentile)
        # loss_delta = train_losses[-2] - train_losses[-1] if len(train_losses) > 1 else 0.0
        # if (len(loss_deltas) > 0) and (loss_delta < (loss_deltas[-1] * early_stop_margin)):

        # early stopping on loss delta
        if (len(train_losses) > 1) and ((train_losses[-2] - train_losses[-1]) < early_stop_margin):

            with torch.no_grad():
                train_losses.append(0.0)
                correct_matches = 0
                y_dim = 0

                for i, (X, y) in enumerate(src_loader):
                    _y_probs = model.forward(X)
                    train_losses[-1] += loss_fn(_y_probs, y).item()
                    _y = _y_probs.softmax(dim=1).argmax(dim=1)
                    correct_matches += (y == _y).sum().item()
                    y_dim += y.size(0)

                train_accs.append(correct_matches / y_dim)
            
            return {
                "model": model,
                "train-losses": train_losses[1:],
                "train-accs": train_accs[1:],
                "val-losses": val_losses,
                "val-accs": val_accs,
            }

        # loss_deltas.append(loss_delta)

    with torch.no_grad():
        train_losses.append(0.0)
        correct_matches = 0
        y_dim = 0

        for i, (X, y) in enumerate(src_loader):
            _y_probs = model.forward(X)
            train_losses[-1] += loss_fn(_y_probs, y).item()
            _y = _y_probs.softmax(dim=1).argmax(dim=1)
            correct_matches += (y == _y).sum().item()
            y_dim += y.size(0)

        train_accs.append(correct_matches / y_dim)
    
    return {
        "model": model,
        "train-losses": train_losses[1:],
        "train-accs": train_accs[1:],
        "val-losses": val_losses,
        "val-accs": val_accs,
    }

r34_sketch = resnet34(pretrained=True)
r34_sketch.fc = Linear(in_features=r34_sketch.fc.in_features, out_features=10)

# model, train_losses, train_accuracies, test_losses, test_accuracies
r34_sketch_metrics = \
    train_and_validate(
        r34_sketch, 
        sketch_train_loader, 
        sketch_test_loader, 
        optimizer=optim.SGD(r34_sketch.parameters(), lr=0.005, momentum=0.9), 
        epochs=100, 
        early_stop_margin=0.1,
        checkpoint_dir_prefix="./checkpoints/sketch-r34"
    )

r34_real = resnet34(pretrained=True)
r34_real.fc = Linear(in_features=r34_real.fc.in_features, out_features=10)

# model, train_losses, train_accuracies, test_losses, test_accuracies
r34_real_metrics = \
    train_and_validate(
        r34_real, 
        real_train_loader, 
        sketch_test_loader, 
        optimizer=optim.SGD(r34_real.parameters(), lr=0.005, momentum=0.9), 
        epochs=100, 
        early_stop_margin=0.1,
        checkpoint_dir_prefix="./checkpoints/real-r34"
    )

plt.clf()
epochs = range(1, len(r34_sketch_metrics["train-losses"]) + 1)

fig, canvas = plt.subplots(1, 1, figsize=(14, 8))

canvas.plot(epochs, r34_sketch_metrics["train-losses"], color=(255/255, 150/255, 69/255), label="Train Losses")
canvas.plot(epochs, r34_sketch_metrics["val-losses"], color=(255/255, 129/255, 131/255), label="Test Losses")
canvas.set_title("Training vs Testing Losses")
canvas.set_xlabel("Epoch")
canvas.set_ylabel("Loss")
canvas.set_xticks(epochs)
canvas.legend(loc="upper right")

plt.savefig("./fig/sketch-r34-losses.jpg")

plt.clf()

fig, canvas = plt.subplots(1, 1, figsize=(14, 8))

canvas.plot(epochs, r34_sketch_metrics["train-accs"], color=(255/255, 150/255, 69/255), label="Train Accuracies")
canvas.plot(epochs, r34_sketch_metrics["val-accs"], color=(255/255, 129/255, 131/255), label="Test Accuracies")
canvas.set_title("Training vs Testing Accuracies")
canvas.set_xlabel("Epoch")
canvas.set_ylabel("Accuracy")

canvas.set_xticks(epochs)
canvas.yaxis.set_major_formatter(PercentFormatter(1.0))
canvas.legend(loc="upper right")

plt.savefig("./fig/sketch-r34-accs.jpg")

plt.clf()
epochs = range(1, len(r34_real_metrics["train-losses"]) + 1)

fig, canvas = plt.subplots(1, 1, figsize=(14, 8))

canvas.plot(epochs, r34_real_metrics["train-losses"], color=(255/255, 150/255, 69/255), label="Train Losses")
canvas.plot(epochs, r34_real_metrics["val-losses"], color=(255/255, 129/255, 131/255), label="Test Losses")
canvas.set_title("Training vs Testing Losses")
canvas.set_xlabel("Epoch")
canvas.set_ylabel("Loss")
canvas.set_xticks(epochs)
canvas.legend(loc="upper right")

plt.savefig("./fig/real-r34-losses.jpg")

plt.clf()

fig, canvas = plt.subplots(1, 1, figsize=(14, 8))

canvas.plot(epochs, r34_real_metrics["train-accs"], color=(255/255, 150/255, 69/255), label="Train Accuracies")
canvas.plot(epochs, r34_real_metrics["val-accs"], color=(255/255, 129/255, 131/255), label="Test Accuracies")
canvas.set_title("Training vs Testing Accuracies")
canvas.set_xlabel("Epoch")
canvas.set_ylabel("Accuracy")

canvas.set_xticks(epochs)
canvas.yaxis.set_major_formatter(PercentFormatter(1.0))
canvas.legend(loc="upper right")

plt.savefig("./fig/real-r34-accs.jpg")