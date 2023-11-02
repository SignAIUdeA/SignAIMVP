from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import cv2
from io import BytesIO
from process_video import predict
app = FastAPI()
import tensorflow as tf



origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frames_list = []

@app.post("/upload/")
async def upload_image(request: Request):
    response = "N/A"
    try:
        # Asume que los datos se envían como JSON y que hay un campo 'image' que contiene la imagen en base64.
        data = await request.json()
        image_data = data.get("image")

        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Aquí puedes procesar, guardar o hacer cualquier operación con la imagen.
        decoded_image = base64.b64decode(image_data.split(",")[-1])
        nparr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames_list.append(frame)
        if len(frames_list) >= 33:
            response = await predict(frames_list[-33:])
            del frames_list[0]
            

        
        # Por ahora, solo vamos a simular la recepción imprimiendo un mensaje.
        #print("Image received!")

        return JSONResponse(content={"message": response}, status_code=200)

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
