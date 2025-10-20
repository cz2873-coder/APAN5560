import torch
import matplotlib.pyplot as plt

def evaluate_model(model, testloader, show_examples=True):
    correct, total = 0, 0
    all_images, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if show_examples:  # 保存一些例子
                all_images.extend(images)
                all_labels.extend(labels)
                all_preds.extend(predicted)

    acc = 100 * correct / total
    print(f"✅ Accuracy: {acc:.2f}%")

    # 🎨 只在最后展示10张预测图
    if show_examples:
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            img = all_images[i].squeeze().numpy()
            label = all_labels[i].item()
            pred = all_preds[i].item()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Pred: {pred} / True: {label}", color="green" if pred == label else "red")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
