from helper_lib.data_loader import get_data_loader
from helper_lib.model import SimpleCNN
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model

trainloader, testloader = get_data_loader()
model = SimpleCNN()
train_model(model, trainloader, epochs=10)
evaluate_model(model, testloader)
save_model(model)
