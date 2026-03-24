from inference import get_model
import os

# This line downloads the weights and saves them to a local folder
# It only uses your API key ONCE to authenticate the download
model = get_model(model_id="accident-vnrxw/2", api_key="qCHzNyNmqVTmbdQhAxWt")

print("Model is now saved locally! You can now code offline.")