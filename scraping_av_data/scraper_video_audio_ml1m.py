import os
import pandas as pd

from yt_dlp import YoutubeDL

# we first manually gathered youtube ids, then we run this script to download the videos

ids = pd.read_csv('ml-youtube.csv')['youtubeId']
download_dir = "download_dir"

ydl_opts = {
    'outtmpl': os.path.join(download_dir, '%(id)s.%(ext)s'),
}

failed_downloads = []

with YoutubeDL(ydl_opts) as ydl:
    for movie_id in ids:
        try:
            file_name = movie_id + '.mp4'
            ydl.download('https://www.youtube.com/watch?v=' + movie_id)
        except Exception:
            failed_downloads.append(movie_id)

print(failed_downloads)
