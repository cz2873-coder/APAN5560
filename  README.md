# Homework 2: CNN Image Classification with FastAPI and Docker

###  Overview
This assignment implements a **Convolutional Neural Network (CNN)** for the MNIST handwritten digits dataset.  
The trained model is deployed using **FastAPI**, and the project is fully containerized with **Docker**.  

The repository follows modular design principles with reusable components inside `helper_lib/`, and an API endpoint `/predict` is available for inference.

---

##  Project Structure
APAN5560_Module4_HelperLib
┣ helper_lib/
┃ ┣ init.py
┃ ┣ data_loader.py
┃ ┣ trainer.py
┃ ┣ model.py
┃ ┣ evaluator.py
┃ ┗ utils.py
┣ app.py
┣ run_mnist.py
┣ requirements.txt
┣ Dockerfile
┗ README.md


---

## ⚙️ Environment Setup

### Option 1: Run locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/APAN5560_Module4_HelperLib.git
   cd APAN5560_Module4_HelperLib
2. Create virtual environment and install dependencies:
   ```bash
     python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
3. Run the training script
   ```bash
   python run_mnist.py
4. Start the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
5. Open your browser and go to:
    ```bash
   http://127.0.0.1:8000/docs


