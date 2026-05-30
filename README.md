# Explainable Multi-Class Chest X-ray Analysis Flask Website


## Project Features

- Upload chest X-ray image
- EfficientNetV2-S prediction
- Four classes: COVID, Lung Opacity, Normal, Viral Pneumonia
- Probability table
- Grad-CAM heatmap
- LLM-style research explanation
- Modern responsive UI

## Folder Structure

```text
cxr_flask_project/
├── app.py
├── config.py
├── model_utils.py
├── gradcam_utils.py
├── explanation_utils.py
├── requirements.txt
├── models/
│   └── best_model.pt        #  trained model here
├── static/
│   ├── css/style.css
│   ├── uploads/
│   └── results/
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    └── about.html
```

## Step 1: Open in VS Code

Open the folder `cxr_flask_project` in VS Code.

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

## Step 4: Add Your Trained Model

Copy your Colab model file into:

```text
models/best_model.pt
```

In Colab, save your trained model using:

```python
torch.save(model.state_dict(), "best_model.pt")
```

Then download `best_model.pt` and place it inside the Flask project's `models/` folder.

## Step 5: Run Website

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

## Important Note

This system is for research and decision-support demonstration only. It is not a substitute for professional medical diagnosis.
