

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
with VideoFileClip(r"D:\Dataset\Double\Videos\Game1.MP4") as video:
    new = video.subclip(0,15)
    new.write_videofile(f"D:\Dataset\Double\Videos\Game1small.mp4", audio_codec='aac')