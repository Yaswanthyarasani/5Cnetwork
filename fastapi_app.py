from fastapi import FastAPI, File, UploadFile
from nested_unet import nested_unet
from attention_unet import attention_unet
import numpy as np
from PIL import Image

app = FastAPI()

model = nested_unet(input_shape=(256, 256, 1))
model.load_weights('model_weights/nested_unet_weights.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert('L').resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).reshape(1, 256, 256, 1)
    prediction = model.predict(img_array)
    return prediction.tolist()
