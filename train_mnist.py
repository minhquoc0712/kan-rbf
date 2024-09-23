"""
train_mnist.py: Train KAN-RBF on MNIST dataset
----------------------------------------------

* Authors: Quoc Nguyen (minhquoc0712@gmail.com)
* Date: 2024-09-23
* Version: 0.0.1

"""

import matplotlib.pyplot as plt
from tqdm import tqdm

import optax
from flax import nnx
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from kan_rbf import KANRBF


def loss_fn(model: nnx.Module, batch):
    logits = model(batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # inplace updates
    optimizer.update(grads)  # inplace updates


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # inplace updates


def load_data(batch_size, drop_last):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch]).numpy()
        labels = torch.tensor([item[1] for item in batch]).numpy()
        images = images.reshape(images.shape[0], -1)
        return {"image": images, "label": labels}

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_set, val_set, train_loader, val_loader


seed = 102
batch_size = 64
drop_last = False
weight_decay = 1e-4
decay_rate = 0.8
layer_size = [28 * 28, 64, 10]
n_epoch = 20
eval_every = 2

train_set, val_set, train_loader, val_loader = load_data(batch_size, drop_last)
n_step_per_epoch = train_set.__len__() // batch_size
if not drop_last:
    n_step_per_epoch += 1

model = KANRBF(layer_size, rngs=nnx.Rngs(seed))
lr = optax.schedules.exponential_decay(
    init_value=1e-3,
    transition_steps=n_step_per_epoch,
    decay_rate=decay_rate,
    staircase=True,
)
optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate=lr, weight_decay=weight_decay)
)
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)

metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}
for epoch in range(n_epoch):
    for step, batch in tqdm(enumerate(train_loader)):
        train_step(model, optimizer, metrics, batch)

    if epoch % eval_every == 0:
        # Log training metrics
        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)
        metrics.reset()  # reset metrics for test set

        # Compute metrics on the test set after each training epoch
        for test_batch in tqdm(val_loader):
            eval_step(model, metrics, test_batch)

        for metric, value in metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)
        metrics.reset()  # reset metrics for next training epoch

        print(
            f"[train] epoch: {epoch}, "
            f"loss: {metrics_history['train_loss'][-1]}, "
            f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
        )
        print(
            f"[test] step: {epoch}, "
            f"loss: {metrics_history['test_loss'][-1]}, "
            f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
        )

plt.figure(figsize=(12, 6))

# Training and Test Loss
plt.subplot(1, 2, 1)
plt.plot(metrics_history["train_loss"], label="Train Loss")
plt.plot(metrics_history["test_loss"], label="Test Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([x * 100 for x in metrics_history["train_accuracy"]], label="Train Accuracy")
plt.plot([x * 100 for x in metrics_history["test_accuracy"]], label="Test Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()
