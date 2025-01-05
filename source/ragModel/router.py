from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ragModel import service

router = APIRouter(prefix="/ragModel")

class QueryRequest(BaseModel):
	query: str

@router.post("/query", response_class=StreamingResponse)
async def query_endpoint(request: QueryRequest):
	return await service.chatBotPipeLine(request.query)