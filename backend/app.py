from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import json
import os
import shutil
from src.data.preprocess import process_single_audio
from src.models.siamese import SiameseNetwork
from src.utils.config import EMBEDDING_DIM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_prediction = ""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
model.load_state_dict(torch.load("models/best_hum_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

@app.post("/")
async def predict_post(file: UploadFile = File(...)):
    global last_prediction
    try:
        temp_wav = f"temp_{file.filename}"
        with open(temp_wav, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        mel, pitch = process_single_audio(temp_wav, duration=16)
        mel = mel.unsqueeze(0).to(device)
        pitch = pitch.unsqueeze(0).to(device)
        
        with torch.no_grad():
            hum_vector = model(mel, pitch)
            
        song_db = torch.load("data/song_index.pt", weights_only=False)
        best_match_name = None
        best_similarity_score = -1.0

        for song_name, chunk_matrix in song_db.items():
            chunk_tensors = torch.tensor(chunk_matrix) 
            similarities = F.cosine_similarity(hum_vector.cpu(), chunk_tensors)
            max_sim_for_song = torch.max(similarities).item()

            if max_sim_for_song > best_similarity_score:
                best_similarity_score = max_sim_for_song
                best_match_name = song_name
                
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        last_prediction = best_match_name
        return {"prediction": best_match_name}
    except Exception as e:
        return {"prediction": "Error"}

@app.get("/")
async def predict_get():
    try:
        return {"prediction": last_prediction}
    except Exception as e:
        return {"prediction": "Error"}