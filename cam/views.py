from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2

def mobile_camera_stream(camera_url):
    cap = cv2.VideoCapture(camera_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@gzip.gzip_page
def live_stream(request):
    camera_url = request.GET.get('camera_url')
    if not camera_url:
        return StreamingHttpResponse("No camera URL provided.")
    
    try:
        return StreamingHttpResponse(mobile_camera_stream(camera_url),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return StreamingHttpResponse(f"Error: {str(e)}")

def index(request):
    camera_url = None
    if request.method == "POST":
        camera_url = request.POST.get('camera_url')
    
    return render(request, 'live_stream.html', {'camera_url': camera_url})


# import cv2
# import numpy as np
# from django.http import StreamingHttpResponse
# from django.shortcuts import render
# from django.views.decorators import gzip
# from ultralytics import YOLO

# # Load YOLO models
# yolo_model = YOLO("yolov10n.pt")  # YOLOv8 model
# fire_model = YOLO("fire_l 1.pt")  # Fire detection model

# # Define labels for YOLOv8 and Fire Detection models
# yolo_labels = yolo_model.names  # YOLOv8 model class names
# fire_labels = fire_model.names  # Fire detection model class names

# def detect_objects(frame):
#     # YOLOv8 object detection
#     results_yolo = yolo_model(frame)
#     # Fire detection
#     results_fire = fire_model(frame)
    
#     # Combine the results
#     combined_results = []
    
#     # Append YOLO results with class offsets
#     for box in results_yolo[0].boxes.data.cpu().numpy():
#         combined_results.append((box, 'yolo'))
    
#     # Append Fire detection results with class offsets
#     for box in results_fire[0].boxes.data.cpu().numpy():
#         combined_results.append((box, 'fire'))
    
#     return combined_results

# def mobile_camera_stream(camera_url):
#     cap = cv2.VideoCapture(camera_url)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Perform object detection on the current frame
#         results = detect_objects(frame)
        
#         # Draw bounding boxes and labels on the frame for every detected object
#         for obj, model_type in results:
#             box, conf, cls = obj[:4], obj[4], int(obj[5])

#             # Check for NaN values
#             if np.isnan(box).any():
#                 print("Detected NaN in bounding box, skipping...")
#                 continue

#             # Assign label text based on the class index
#             if model_type == 'yolo' and cls < len(yolo_labels):
#                 label_text = yolo_labels[cls]
#                 color = (0, 255, 0)  # Green for YOLOv8 detections
#             elif model_type == 'fire' and cls < len(fire_labels):
#                 label_text = fire_labels[cls]
#                 color = (0, 0, 255)  # Red for Fire detection
#             else:
#                 label_text = f'Unknown Class {cls}'
#                 color = (255, 255, 255)  # White for unknown classes
            
#             # Draw bounding box
#             cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            
#             # Draw label text above the bounding box
#             cv2.putText(frame, f'{label_text}: {conf:.2f}', (int(box[0]), int(box[1]) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         _, jpeg = cv2.imencode('.jpg', frame)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
#     cap.release()

# @gzip.gzip_page
# def live_stream(request):
#     camera_url = request.GET.get('camera_url')
#     if not camera_url:
#         return StreamingHttpResponse("No camera URL provided.")
    
#     try:
#         return StreamingHttpResponse(mobile_camera_stream(camera_url),
#                                      content_type='multipart/x-mixed-replace; boundary=frame')
#     except Exception as e:
#         return StreamingHttpResponse(f"Error: {str(e)}")

# def index(request):
#     camera_url = None
#     if request.method == "POST":
#         camera_url = request.POST.get('camera_url')
#     return render(request, 'live_stream.html', {'camera_url': camera_url})
 
####################################################################################
import os
from django.http import StreamingHttpResponse
from django.conf import settings

def stream_flv(request):
    flv_file_path = f'{settings.BASE_DIR}/feed.flv'  # Update with the correct path

    def stream_video():
        with open(flv_file_path, 'rb') as flv_file:
            while True:
                chunk = flv_file.read(8192)  # Read in chunks
                if not chunk:
                    break
                yield chunk
        os.remove(flv_file_path)  # Remove the file after streaming

    response = StreamingHttpResponse(stream_video(), content_type='video/x-flv')
    response['Content-Disposition'] = 'inline; filename="live_feed.flv"'
    return response


def view_stream(request):
    return render(request, 'stream_flv.html')
