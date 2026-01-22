import os
import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import json

# Инициализация приложения
app = FastAPI(title="Helmet Detection API")

# Разрешаем запросы с любого источника (для тестирования)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель при запуске
print("🚀 Загружаю модель YOLO...")
model = YOLO('best_model.pt')  # Файл должен быть в той же папке
print("✅ Модель загружена!")

def process_image(image_bytes):
    """Обрабатывает изображение и возвращает результат"""
    # Преобразуем bytes в numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Конвертируем BGR в RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Детекция с помощью YOLO
    results = model(img_rgb, conf=0.25)  # порог уверенности 25%
    
    # Получаем первый результат (т.к. одно изображение)
    result = results[0]
    
    # Подсчитываем статистику
    helmet_count = 0
    no_helmet_count = 0
    
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 1:  # helmet
                helmet_count += 1
            else:  # no_helmet (0)
                no_helmet_count += 1
    
    # Рисуем bounding boxes на изображении
    annotated_img = result.plot()  # YOLO сам рисует боксы
    
    # Конвертируем обратно в BGR для сохранения
    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    
    return {
        'image': annotated_img_bgr,
        'helmet_count': helmet_count,
        'no_helmet_count': no_helmet_count,
        'total_people': helmet_count + no_helmet_count,
        'with_helmet_percentage': (helmet_count / (helmet_count + no_helmet_count) * 100) if (helmet_count + no_helmet_count) > 0 else 0
    }

@app.get("/")
async def root():
    return {"message": "Helmet Detection API работает!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Принимает изображение, возвращает JSON с результатами"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Читаем файл
    contents = await file.read()
    
    # Обрабатываем изображение
    result = process_image(contents)
    
    # Возвращаем JSON с результатами
    return {
        "helmet_count": result['helmet_count'],
        "no_helmet_count": result['no_helmet_count'],
        "total_people": result['total_people'],
        "with_helmet_percentage": round(result['with_helmet_percentage'], 1),
        "has_helmets": result['helmet_count'] > 0,
        "all_with_helmets": result['helmet_count'] > 0 and result['no_helmet_count'] == 0
    }

@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    """Принимает изображение, возвращает обработанное изображение с боксами"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Читаем файл
    contents = await file.read()
    
    # Обрабатываем изображение
    result = process_image(contents)
    
    # Конвертируем изображение в bytes
    _, encoded_img = cv2.imencode('.jpg', result['image'])
    
    # Возвращаем изображение
    return StreamingResponse(
        io.BytesIO(encoded_img.tobytes()),
        media_type="image/jpeg",
        headers={
            "helmet-count": str(result['helmet_count']),
            "no-helmet-count": str(result['no_helmet_count'])
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)