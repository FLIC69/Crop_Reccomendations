from fastapi import FastAPI, Request, logger
from app.routers import predict

app = FastAPI()

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    sanitized_input = {k: "****" if k == "password" else v 
                      for k,v in request.items()}
    logger.info(f"{request.url.path} - {sanitized_input}")
    return response

app.include_router(predict.router, prefix="/predict")

@app.get("/")
def home():
    return "Healthy"