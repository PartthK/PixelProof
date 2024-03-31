import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filename, extension = os.path.splitext(event.src_path)
        if extension.lower() in ['.mp4', '.avi', '.mkv', '.mov']:  # Adjust the list of supported video extensions as needed
            print(f"New video file detected: {event.src_path}")
            subprocess.run(["python", "both.py"])

if __name__ == "__main__":
    folder_to_watch = "videos"
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=True)
    observer.start()
    print(f"Watching folder '{folder_to_watch}' for new video files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()