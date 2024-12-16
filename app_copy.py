# import asyncio
# import os
# import logging
# import subprocess
# import shutil
# from datetime import datetime

# from pyrtmp import StreamClosedException
# from pyrtmp.flv import FLVFileWriter, FLVMediaType
# from pyrtmp.session_manager import SessionManager
# from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


# class RTMP2MP4Controller(SimpleRTMPController):

#     def __init__(self, output_directory: str):
#         self.output_directory = output_directory
#         self.start_time = None
#         self.chunk_duration = 20  # Duration of each chunk in seconds (20 seconds)
#         self.chunk_counter = 0  # Start from the first chunk
#         self.session_active = False  # Track if the session is active
#         super().__init__()

#     async def on_ns_publish(self, session, message) -> None:
#         self.chunk_counter = 0  # Reset chunk counter for new stream
#         publishing_name = message.publishing_name
#         flv_file_path = os.path.join(self.output_directory, "feed.flv")
#         session.state = FLVFileWriter(output=flv_file_path)
#         session.flv_file_path = flv_file_path  # Store FLV file path in session state
#         self.start_time = datetime.now()  # Initialize start time
#         self.session_active = True  # Mark session as active
#         asyncio.create_task(self.convert_flv_to_mp4_in_chunks(session))  # Start conversion task
#         await super().on_ns_publish(session, message)

#     async def on_metadata(self, session, message) -> None:
#         session.state.write(0, message.to_raw_meta(), FLVMediaType.OBJECT)
#         await super().on_metadata(session, message)

#     async def on_video_message(self, session, message) -> None:
#         session.state.write(message.timestamp, message.payload, FLVMediaType.VIDEO)
#         await super().on_video_message(session, message)

#     async def on_audio_message(self, session, message) -> None:
#         session.state.write(message.timestamp, message.payload, FLVMediaType.AUDIO)
#         await super().on_audio_message(session, message)

#     async def on_stream_closed(self, session: SessionManager, exception: StreamClosedException) -> None:
#         self.session_active = False  # Mark session as inactive
#         session.state.close()  # Close the FLV file
#         await super().on_stream_closed(session, exception)

#     async def convert_flv_to_mp4_in_chunks(self, session):
#         """Convert FLV to MP4 in chunks."""
#         while self.session_active:
#             # Calculate the time elapsed since the start of the stream
#             elapsed_time = (datetime.now() - self.start_time).total_seconds()

#             # If the elapsed time is greater than the chunk duration, convert the FLV to MP4
#             if elapsed_time >= self.chunk_duration:
#                 self.chunk_counter += 1
#                 chunk_name = f"feed_clip_{self.chunk_counter}.mp4"
#                 chunk_mp4_file_path = os.path.join(self.output_directory, "videos", chunk_name)

#                 # Convert the full FLV file to the main MP4 file and the current chunk
#                 await self.convert_to_mp4(session.flv_file_path, "feed.mp4", chunk_mp4_file_path)

#                 # Manage chunk files: keep the latest 10 and delete older ones
#                 await self.manage_chunk_files()

#                 # Update start time for the next chunk
#                 self.start_time = datetime.now()

#             # Wait for a short period before checking again
#             await asyncio.sleep(1)

#     async def convert_to_mp4(self, flv_file_path, main_mp4_file_path, chunk_mp4_file_path):
#         """Convert FLV file to MP4 using ffmpeg."""
#         # Convert the FLV file to the main MP4 file
#         main_command = [
#             'ffmpeg',
#             '-i', flv_file_path,  # Input FLV file
#             '-c:v', 'copy',       # Copy the video stream without re-encoding
#             '-c:a', 'aac',        # Convert the audio stream to AAC
#             '-strict', 'experimental',
#             os.path.join(self.output_directory, main_mp4_file_path)  # Output main MP4 file
#         ]

#         # Convert the FLV file to the chunk MP4 file
#         chunk_command = [
#             'ffmpeg',
#             '-i', flv_file_path,  # Input FLV file
#             '-c:v', 'copy',       # Copy the video stream without re-encoding
#             '-c:a', 'aac',        # Convert the audio stream to AAC
#             '-strict', 'experimental',
#             chunk_mp4_file_path   # Output chunk MP4 file
#         ]

#         try:
#             subprocess.run(main_command, check=True)
#             subprocess.run(chunk_command, check=True)
#             logger.info(f"Converted {flv_file_path} to {chunk_mp4_file_path}")

#             # Move the chunk MP4 file to /var/www/html/videos
#             destination_path = f'/var/www/html/videos/{os.path.basename(chunk_mp4_file_path)}'
#             shutil.move(chunk_mp4_file_path, destination_path)
#             logger.info(f"Moved chunk MP4 file to {destination_path}")

#         except subprocess.CalledProcessError as e:
#             logger.error(f"Failed to convert {flv_file_path} to MP4: {e}")
#         except Exception as e:
#             logger.error(f"Failed to move MP4 file: {e}")

#     async def manage_chunk_files(self):
#         """Keep only the latest 10 chunk files and delete older ones."""
#         video_dir = os.path.join(self.output_directory, "videos")
#         chunk_files = sorted(
#             [f for f in os.listdir(video_dir) if f.startswith("feed_clip_")],
#             key=lambda f: int(f.split("_")[2].split(".")[0])
#         )

#         if len(chunk_files) > 10:
#             # Delete the oldest chunk files beyond the latest 10
#             for file_to_delete in chunk_files[:-10]:
#                 file_path = os.path.join(video_dir, file_to_delete)
#                 try:
#                     os.remove(file_path)
#                     logger.info(f"Deleted old chunk file: {file_path}")
#                 except Exception as e:
#                     logger.error(f"Failed to delete old chunk file {file_path}: {e}")


# class SimpleServer(SimpleRTMPServer):

#     def __init__(self, output_directory: str):
#         self.output_directory = output_directory
#         super().__init__()

#     async def create(self, host: str, port: int):
#         loop = asyncio.get_event_loop()
#         self.server = await loop.create_server(
#             lambda: RTMPProtocol(controller=RTMP2MP4Controller(self.output_directory)),
#             host=host,
#             port=port,
#         )


# async def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     videos_dir = os.path.join(current_dir, "videos")

#     # Create the videos directory if it doesn't exist
#     os.makedirs(videos_dir, exist_ok=True)

#     server = SimpleServer(output_directory=current_dir)
#     await server.create(host='0.0.0.0', port=1935)
#     await server.start()
#     await server.wait_closed()


# if __name__ == "__main__":
#     asyncio.run(main())
# #ok

import asyncio
import os
import logging
import subprocess
import shutil
from datetime import datetime
import cv2
from ultralytics import YOLO
from pyrtmp import StreamClosedException
from pyrtmp.flv import FLVFileWriter, FLVMediaType
from pyrtmp.session_manager import SessionManager
from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_fire_detection_model():
    model_path = os.path.join("yolov10n.pt")
    if os.path.exists(model_path):
        logging.debug("Fire detection model loaded successfully.")
        return YOLO(model_path)
    logging.error("Fire detection model not found.")
    return None

def process_frame(model, frame, conf_threshold, iou_threshold):
    combined_frame = frame.copy()
    res = model.predict(combined_frame, conf=conf_threshold, iou=iou_threshold, device='cpu')
    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1
    latency = round(sum(res[0].speed.values()) / 1000, 10)
    prediction_text = "Predicted "
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}(s), ' if v > 1 else f'{v} {k}, '
    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"
    prediction_text += f' in {latency} seconds.'
    combined_frame = cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB)
    return combined_frame, prediction_text

async def detect_and_save(video_file_path, conf_threshold=0.20, iou_threshold=0.5, frame_skip=2):
    model = load_fire_detection_model()
    if model is None:
        raise Exception("Fire Detection model not found")

    conf_threshold = float(conf_threshold)
    iou_threshold = float(iou_threshold)

    video = cv2.VideoCapture(video_file_path)
    if not video.isOpened():
        raise Exception("Could not open video file")

    frame_rate = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    detected_output_file_path = os.path.splitext(video_file_path)[0] + "_detected.mp4"
    out = cv2.VideoWriter(detected_output_file_path, fourcc, frame_rate, (width, height))

    frame_count = 0
    last_processed_frame = None

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            processed_frame, _ = process_frame(model, frame, conf_threshold, iou_threshold)
            last_processed_frame = processed_frame
        else:
            processed_frame = last_processed_frame
        
        out.write(processed_frame)
        frame_count += 1
    
    video.release()
    out.release()
    
    reencoded_output_file_path = os.path.splitext(detected_output_file_path)[0] + "_reencoded.mp4"
    convert_mp4_to_mp4(detected_output_file_path, reencoded_output_file_path)

    return reencoded_output_file_path

def convert_mp4_to_mp4(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-movflags', '+faststart',
        output_file
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"Converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {input_file} to MP4: {e}")

class RTMP2MP4Controller(SimpleRTMPController):
    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.start_time = None
        self.chunk_duration = 20
        self.chunk_counter = 0
        self.session_active = False
        super().__init__()

    async def on_ns_publish(self, session, message) -> None:
        self.chunk_counter = 0
        flv_file_path = os.path.join(self.output_directory, "feed.flv")
        session.state = FLVFileWriter(output=flv_file_path)
        session.flv_file_path = flv_file_path
        self.start_time = datetime.now()
        self.session_active = True
        asyncio.create_task(self.convert_flv_to_mp4_in_chunks(session))
        await super().on_ns_publish(session, message)

    async def on_metadata(self, session, message) -> None:
        session.state.write(0, message.to_raw_meta(), FLVMediaType.OBJECT)
        await super().on_metadata(session, message)

    async def on_video_message(self, session, message) -> None:
        session.state.write(message.timestamp, message.payload, FLVMediaType.VIDEO)
        await super().on_video_message(session, message)

    async def on_audio_message(self, session, message) -> None:
        session.state.write(message.timestamp, message.payload, FLVMediaType.AUDIO)
        await super().on_audio_message(session, message)

    async def on_stream_closed(self, session: SessionManager, exception: StreamClosedException) -> None:
        self.session_active = False
        session.state.close()
        await super().on_stream_closed(session, exception)

    async def convert_flv_to_mp4_in_chunks(self, session):
        while self.session_active:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if elapsed_time >= self.chunk_duration:
                self.chunk_counter += 1
                chunk_name = f"feed_clip_{self.chunk_counter}.mp4"
                chunk_mp4_file_path = os.path.join(self.output_directory, "videos", chunk_name)
                await self.convert_to_mp4(session.flv_file_path, "feed.mp4", chunk_mp4_file_path)
                await detect_and_save(chunk_mp4_file_path)
                await self.manage_chunk_files()
                self.start_time = datetime.now()
            await asyncio.sleep(1)

    async def convert_to_mp4(self, flv_file_path, main_mp4_file_path, chunk_mp4_file_path):
        main_command = [
            'ffmpeg',
            '-i', flv_file_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            os.path.join(self.output_directory, main_mp4_file_path)
        ]
        chunk_command = [
            'ffmpeg',
            '-i', flv_file_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            chunk_mp4_file_path
        ]
        try:
            subprocess.run(main_command, check=True)
            subprocess.run(chunk_command, check=True)
            destination_path = os.path.join(self.output_directory, "videos", os.path.basename(chunk_mp4_file_path))
            shutil.move(chunk_mp4_file_path, destination_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {flv_file_path} to MP4: {e}")
        except Exception as e:
            logger.error(f"Failed to move MP4 file: {e}")

    async def manage_chunk_files(self):
        video_dir = os.path.join(self.output_directory, "videos")
        chunk_files = sorted(
            [f for f in os.listdir(video_dir) if f.startswith("feed_clip_")],
            key=lambda f: int(f.split("_")[2].split(".")[0])
        )
        if len(chunk_files) > 10:
            for file_to_delete in chunk_files[:-10]:
                file_path = os.path.join(video_dir, file_to_delete)
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old chunk file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete old chunk file {file_path}: {e}")

class SimpleServer(SimpleRTMPServer):
    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        super().__init__()

    async def create(self, host: str, port: int):
        loop = asyncio.get_event_loop()
        self.server = await loop.create_server(
            lambda: RTMPProtocol(controller=RTMP2MP4Controller(self.output_directory)),
            host=host,
            port=port,
        )

async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(current_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    server = SimpleServer(output_directory=current_dir)
    await server.create(host='0.0.0.0', port=1935)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
