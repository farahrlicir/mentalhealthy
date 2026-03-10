from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

# Load model and tokenizer once at startup
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.config.hidden_size, 6)
)
model.load_state_dict(torch.load('best_model_dropout02_distilbert.pth', map_location='cpu'))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Your class map
class_map = {
    0: "adhd",
    1: "anxiety",
    2: "bipolar",
    3: "depression",
    4: "ptsd",
    5: "none"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=500)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        predicted_class_name = class_map.get(predicted_class_id, "Unknown")

    return jsonify({
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": predicted_class_name,
        "probabilities": probs.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
