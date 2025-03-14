import torch
import cv2
import time
import threading
import queue
import pyttsx3
import requests
from googletrans import Translator
from datetime import datetime
import pytesseract
import signal
# Global variable to manage the application's running state
running = True
def signal_handler(sig, frame):
    global running
    running = False
    print("\nShutting down gracefully...")
signal.signal(signal.SIGINT, signal_handler)
def signal_handler(sig, frame):
    global running
    running = False
    print("\nShutting down gracefully...")
signal.signal(signal.SIGINT, signal_handler)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', skip_validation=True)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
announcement_queue = queue.Queue()
translator = Translator()
def process_announcements(language='en'):
    while running:
        try:
            announcement = announcement_queue.get(timeout=1)
            if announcement == "STOP":
                break
            print(f"Announcing: {announcement}")
            if language != 'en':
                translated_text = translator.translate(announcement, src='en', dest=language).text
                print(f"Translated Announcement: {translated_text}")
                engine.say(translated_text)
            else:
                engine.say(announcement)
            engine.runAndWait()
            announcement_queue.task_done()
        except queue.Empty:
            continue
def add_to_announcement_queue(objects):
    for obj in objects:
        announcement_queue.put(obj)
def get_object_spatial_info(frame_width, bbox):
    x_center = (bbox[0] + bbox[2]) // 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if x_center < frame_width * 0.33:
        position = "left"
    elif x_center > frame_width * 0.66:
        position = "right"
    else:
        position = "center"
    distance = round(10000 / (width * height), 1)
    return position, distance
def get_weather():
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"
    city = "Bangalore"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            temp = data['main']['temp']
            weather_desc = data['weather'][0]['description']
            return f"The temperature is {temp}°C with {weather_desc}."
        else:
            return "Could not fetch weather information."
    except Exception as e:
        return "Weather service is unavailable."
def select_language():
    languages = {"1": "en", "2": "kn", "3": "hi"}
    print("Select a language for announcements:")
    print("1: English")
    print("2: Kannada")
    print("3: Hindi")
    choice = input("Enter the number corresponding to your choice: ")
    return languages.get(choice, "en")
def process_text_extraction(frame, language):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(threshold_frame, lang='eng')
    print(f"Extracted Text: {text}")
    cv2.imshow("Processed Frame for OCR", threshold_frame)
    cv2.waitKey(1)
    if text.strip():
        if language != 'en':
            translated_text = translator.translate(text, src='en', dest=language).text
            print(f"Translated Text: {translated_text}")
            engine.say(translated_text)
        else:
            engine.say(text)
        engine.runAndWait()
    else:
        print("No text detected.")
def main():
    global running
    language = select_language()
    announcement_thread = threading.Thread(target=process_announcements, args=(language,), daemon=True)
    announcement_thread.start()
    weather_info = get_weather()
    date_time_info = announce_date_time()
    add_to_announcement_queue([date_time_info, weather_info])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cooldown_period = 5
    last_announcement_time = time.time()
    last_text_extraction_time = time.time()
    text_extraction_cooldown = 3
    announced_objects = set()
    mode = "object_detection"
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_height, frame_width, _ = frame.shape
        if mode == "object_detection":
            result = model(frame)
            data_frame = result.pandas().xyxy[0]
            detected_objects = []
            for index in data_frame.index:
                x1, y1, x2, y2 = map(int, [
                    data_frame['xmin'][index],
                    data_frame['ymin'][index],
                    data_frame['xmax'][index],
                    data_frame['ymax'][index]
                ])
                label = data_frame['name'][index]
                conf = data_frame['confidence'][index]
                position, distance = get_object_spatial_info(frame_width, (x1, y1, x2, y2))
                object_key = f"{label}{position}{distance}"
                if object_key not in announced_objects:
                    detected_objects.append(f"{label} on the {position}, {distance} meters away")
                    announced_objects.add(object_key)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            if detected_objects and time.time() - last_announcement_time > cooldown_period:
                add_to_announcement_queue(detected_objects)
                last_announcement_time = time.time()
            cv2.putText(frame, "Press 't' for text extraction mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif mode == "text_extraction":
            if time.time() - last_text_extraction_time > text_extraction_cooldown:
                threading.Thread(target=process_text_extraction, args=(frame, language), daemon=True).start()
                last_text_extraction_time = time.time()
            cv2.putText(frame, "Press 'o' for object detection mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-Time Processing', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            mode = "text_extraction"
        elif key == ord('o'):
            mode = "object_detection"
    running = False
    announcement_queue.put("STOP")
    announcement_thread.join()
    cap.release()
    cv2.destroyAllWindows()
if _name_ == '_main_':
    main()
