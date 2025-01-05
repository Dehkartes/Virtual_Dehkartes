from fastapi import FastAPI, Request, status
from fastapi.responses import Response
from allowIP import ALLOWED_IPS

from ragModel.router import router as ragRouter

import uvicorn
import warnings

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

app = FastAPI()

app.include_router(ragRouter)

@app.middleware("http")
async def ip_filter_middleware(request: Request, call_next):
	client_ip = request.client.host
	if client_ip not in ALLOWED_IPS:
		return Response(status_code=status.HTTP_403_FORBIDDEN)
	response = await call_next(request)
	return response

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)