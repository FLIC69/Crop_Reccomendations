from fastapi import FastAPI, Request, logger
from app.routers import predict
from app.utils import logger

app = FastAPI()

app.add_middleware(logger.LoggingMiddleware)

app.include_router(predict.router, prefix="/predict")

@app.get("/")
def home():
    return "Healthy"