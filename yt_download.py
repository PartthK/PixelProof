from pytube import YouTube
from moviepy.editor import *

# download the video
yt_video = YouTube('https://www.youtube.com/shorts/kUp46qUob60').streams.get_lowest_resolution()
yt_video.download(output_path='/Users/davidstonestreet/Desktop/Purdue/Catapult AI Hackathon 2024/PixelProof/videos', filename="video.mp4")

# ignore this, this downloads only a cut portion of the full video
video = VideoFileClip("videos/video.mp4").subclip(1,10)
video.write_videofile("videos/clip.mp4")