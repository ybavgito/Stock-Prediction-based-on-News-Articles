import torch
from torch import nn
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class FinBERTRegressor(nn.Module):
    def __init__(self, finbert_model_name="ProsusAI/finbert", hidden_size=768, output_size=1):
        super(FinBERTRegressor, self).__init__()
        # Load the FinBERT model
        self.finbert = BertModel.from_pretrained(finbert_model_name)

        # Regression layer
        self.regressor = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get embeddings from FinBERT
        with torch.no_grad():  # No need to calculate gradients for FinBERT
            outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)

        # We use the [CLS] token's embedding for regression
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Pass through the regression layer
        return self.regressor(cls_embedding)

# Initialize model and tokenizer
try:
    model = FinBERTRegressor()
    model.to(device)
    # Load model weights explicitly on CPU
    model.regressor.load_state_dict(torch.load('model_regressor_weight_v2.pth', map_location=torch.device('cpu')))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

try:
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    raise

def predict_monthly_stock_price_change_rate(text: str):
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Move the inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            output = model(**inputs)

        return output.item()
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise
