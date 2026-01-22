from ultralytics import YOLO
import cv2
import numpy as np
import sys

print("🔍 ТЕСТ КЛАССОВ МОДЕЛИ")
print("="*50)

# Загрузите модель
try:
    model = YOLO('best_model.pt')
    print("✅ Модель загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    sys.exit(1)

# Спросим у пользователя путь к тестовому изображению
test_image_path = input("Введите путь к тестовому изображению (или нажмите Enter для тестовой картинки): ")

if test_image_path and test_image_path.strip():
    # Загружаем изображение пользователя
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"❌ Не удалось загрузить изображение: {test_image_path}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✅ Загружено изображение: {test_image_path}")
else:
    # Создаем тестовое изображение с человеком БЕЗ каски
    print("🖼️ Создаю тестовое изображение с человеком БЕЗ каски...")
    img = np.zeros((640, 640, 3), dtype=np.uint8) + 220  # светлый фон
    
    # Рисуем простого человека (красный = без каски)
    # Тело
    cv2.rectangle(img, (250, 150), (390, 400), (255, 0, 0), -1)  # синее тело
    # Голова (круг, без каски)
    cv2.circle(img, (320, 120), 50, (200, 100, 0), -1)  # коричневая голова
    # Сохраним для просмотра
    cv2.imwrite('test_person_no_helmet.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("✅ Создано тестовое изображение: test_person_no_helmet.jpg")

print("\n🧠 Запускаю детекцию...")

# Детекция
results = model(img, conf=0.3)

print(f"\n📊 РЕЗУЛЬТАТЫ:")
print("-"*30)

if results[0].boxes is not None and len(results[0].boxes) > 0:
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        print(f"\n👤 Объект {i+1}:")
        print(f"   Class ID: {class_id}")
        print(f"   Уверенность: {confidence:.2%}")
        
        # Что означает каждый класс согласно dataset.yaml?
        print(f"   Согласно dataset.yaml:")
        print(f"     0 = 'no_helmet' (без каски)")
        print(f"     1 = 'helmet' (с каской)")
        
        # Интерпретация
        if class_id == 0:
            print(f"   🧐 Модель считает: 'no_helmet' (БЕЗ КАСКИ)")
        else:
            print(f"   🧐 Модель считает: 'helmet' (С КАСКОЙ)")
            
        # Координаты bounding box
        bbox = box.xyxy[0].cpu().numpy()
        print(f"   📏 BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    print("\n" + "="*50)
    print("🤔 ВЫВОД:")
    
    # Проверяем логику
    if "без каски" in test_image_path.lower() or "no_helmet" in test_image_path.lower():
        print("Вы тестировали изображение БЕЗ каски.")
        print("Если модель сказала class_id=0 → ВСЁ ПРАВИЛЬНО")
        print("Если модель сказала class_id=1 → КЛАССЫ ПЕРЕВЕРНУТЫ")
    else:
        print("По названию непонятно, что на изображении.")
        print("Нужно понять по логике:")
        print("- class_id=0 должен быть для людей БЕЗ касок")
        print("- class_id=1 должен быть для людей С касками")
        
else:
    print("❌ Модель не обнаружила ни одного объекта!")

# Сохраним результат для визуальной проверки
annotated = results[0].plot()
cv2.imwrite('test_result.jpg', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
print(f"\n💾 Результат сохранен: test_result.jpg")
print("Откройте этот файл, чтобы увидеть, как модель разметила изображение.")

# Дополнительный тест с разными порогами уверенности
print("\n🔬 ТЕСТ С РАЗНЫМИ ПОРОГАМИ:")
for conf_threshold in [0.1, 0.25, 0.5, 0.7]:
    results_conf = model(img, conf=conf_threshold)
    if results_conf[0].boxes is not None:
        print(f"conf={conf_threshold}: обнаружено {len(results_conf[0].boxes)} объектов")
    else:
        print(f"conf={conf_threshold}: объектов не обнаружено")