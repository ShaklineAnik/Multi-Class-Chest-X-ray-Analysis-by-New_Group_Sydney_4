import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# Same normalization fallback used in your Colab notebook.
# Change these if your notebook printed different mean/std values.
NORM_MEAN = 0.5
NORM_STD = 0.25

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# Keep False for normal VS Code CPU use. The app still generates a strong template explanation.
USE_LOCAL_LLM = False
