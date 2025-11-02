APAN5560-homework3-GAN/
│── app.py
│── main.py
│── Dockerfile
│── requirements.txt
│
└── helper_lib/
    ├── model.py
    ├── trainer.py
    └── generator.py

---

## **How to Run the FastAPI Server (Locally)**

1. Install dependencies:
pip install -r requirements.txt
2. Start the FastAPI server:
uvicorn app:app --host 0.0.0.0 --port 8000

3. Open in browser:
http://127.0.0.1:8000

4. API documentation:

---

##  **API Endpoints**

### **1. GET /**  
Check if the server is running.

### **2. POST /gan/generate**  
Generates a **3×3 grid PNG** of GAN-generated MNIST digits.  
Returns `image/png` directly.

**Example response:**  
A real generated image (gan_grid_3x3.png) is included in the repository.

---

## **Training the Model**

To retrain the GAN locally:

