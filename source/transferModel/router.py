from fastapi import APIRouter

router = APIRouter(prefix="/transferModel")

@router.get("/hello")
async def read_hello():
	return {"message": "Hello, world!"}