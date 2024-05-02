from fastapi import FastAPI, UploadFile, File
from main import main_func
from typing import List
from pydantic import BaseModel


class VideoItem(BaseModel):
    VideoTitle: str


app = FastAPI()


@app.get("/")
def home():
    return {'message': "Welcome to the home page"}


@app.post("/predict")
def predict(video: VideoItem):
    #title = video.VideoTitle
    results = main_func(video.VideoTitle)
    return {'message': results}







