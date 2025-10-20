import torch

def save_model(model, path='cnn_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")

def load_model(model, path='cnn_model.pth'):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f"Model loaded from {path}")
    return model
