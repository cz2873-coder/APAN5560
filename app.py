from fastapi import FastAPI, UploadFile
from helper_lib.model import SimpleCNN
from helper_lib.utils import save_model
from helper_lib.evaluator import evaluate_model
from helper_lib.data_loader import get_data_loader
import torch

app = FastAPI()

@app.get("/")
def home():
    return {"message": "MNIST CNN API is running"}

@app.get("/train")
def train():
    trainloader, testloader = get_data_loader()
    model = SimpleCNN()
    from helper_lib.trainer import train_model
    train_model(model, trainloader, epochs=3)
    torch.save(model.state_dict(), "cnn_model.pth")
    return {"status": "Model trained successfully"}

@app.post("/predict")
async def predict(file: UploadFile):
    # 读取上传图片并预测类别
    from PIL import Image
    import torchvision.transforms as transforms
    model = SimpleCNN()
    model.load_state_dict(torch.load("cnn_model.pth"))
    model.eval()
    image = Image.open(file.file).convert("L")
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    predicted = torch.argmax(output, 1).item()
    return {"prediction": predicted}
