import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from logging.handlers import RotatingFileHandler
import json
import os
from typing import Callable
import time

class AppLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)

        self.access_logger = logging.getLogger("app.access")
        self.error_logger = logging.getLogger("app.error")

        self._configure_logger(self.access_logger, f"{log_dir}/access.log", logging.INFO)
        self._configure_logger(self.error_logger, f"{log_dir}/error.log", logging.ERROR)

    def _configure_logger(self, logger, file_path, level):
        logger.setLevel(level)
        if not logger.handlers:
            handler = RotatingFileHandler(file_path, maxBytes=10**6, backupCount=5)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

    def log_access(self, message):
        self.access_logger.info(message)

    def log_error(self, message):
        self.error_logger.error(message)

logger = AppLogger()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        client_ip = request.client.host
        method = request.method
        url = str(request.url)

        # Basic request logging without body
        log_message = f"Request from {client_ip}: {method} {url}"
        logger.log_access(log_message)

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.log_access(
                f"Response to {client_ip}: {response.status_code} "
                f"| Processing time: {process_time:.4f}s"
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.log_error(
                f"Exception for {client_ip} after {process_time:.4f}s: {str(e)}"
            )
            raise


# Alternative: Middleware that handles body logging at the application level
class BodyLoggingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Extract request info
        method = scope.get("method", "")
        path = scope.get("path", "")
        client = scope.get("client", ["unknown", 0])
        client_ip = client[0] if client else "unknown"
        
        # Log basic request info
        logger.log_access(f"Request from {client_ip}: {method} {path}")
        
        # Handle body logging for POST/PUT/PATCH
        if method in ("POST", "PUT", "PATCH"):
            body_parts = []
            
            async def logging_receive():
                message = await receive()
                if message["type"] == "http.request":
                    body_parts.append(message.get("body", b""))
                    
                    # If this is the last chunk, log the body
                    if not message.get("more_body", False):
                        full_body = b"".join(body_parts)
                        if full_body:
                            try:
                                data = json.loads(full_body)
                                logger.log_access(f"Payload for {client_ip}: {json.dumps(data)}")
                            except json.JSONDecodeError:
                                logger.log_access(f"Raw payload for {client_ip}: {full_body.decode('utf-8', 'ignore')[:500]}")
                
                return message
            
            await self.app(scope, logging_receive, send)
        else:
            await self.app(scope, receive, send)