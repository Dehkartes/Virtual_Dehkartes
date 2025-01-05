from fastapi.responses import StreamingResponse
from ragModel import model

async def chatBotPipeLine(query: str):
	return StreamingResponse(model.pipeLine(query), media_type="text/plain")