
from yt_dlp import YoutubeDL
import os 
import re
import whisper
import torch
import json

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# To run whisper, install ffmpeg, add to PATH, link: https://www.ffmpeg.org/download.html

class YoutubeAudioDownloader:
    def __init__(self, output_folder):
        self.output_folder = os.path.abspath(output_folder)
        self.audio_files_dict = {}

    def get_safe_filename(self, filename):
        # replace any character in the input filename that is not a word character, a hyphen, or a dot with an underscore
        safe_filename = re.sub(r'^\w\-.:', '_', filename)
        # replace full-width colon with an underscore
        safe_filename = re.sub(r'\：', '_', safe_filename)
        # replace space with an underscore
        safe_filename = re.sub(r' ', '_', safe_filename)
        # replace consecutive _ with a single _
        safe_filename = re.sub(r'_+', '_', safe_filename)
        # truncate to max of 50 characters, and remove remaining underscore
        # safe_filename = safe_filename[:50].strip('_')

        return safe_filename
    
    def download_audio(self, video_url):
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(self.output_folder, '%(title)s.%(ext)s'),
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                filename = ydl.prepare_filename(info)
                base, ext = os.path.splitext(filename)
                base = self.get_safe_filename(base)
                new_file = base + '.mp3'

            print(f"Audio file downloaded: {new_file}")
            self.audio_files_dict[video_url] = new_file
            return new_file
        except Exception as e:
            print(f"Error downloading audio from {video_url}: {str(e)}")
            return None
        
    def download_multiple_audios(self, video_urls):
        for url in video_urls:
            print(f"Processing video: {url}")
            audio_file = self.download_audio(url)
            if audio_file is None:
                print(f"Failed to download audio from: {url}")
        
        return self.audio_files_dict

class AudioTranscriber:
    def __init__(self, input_folder, whisper_model):
        self.input_folder = os.path.abspath(os.path.join(os.getcwd(), input_folder))
        self.whisper_model = whisper_model
        self.transcriptions_dict = {}
    
    def transcribe_audio(self, audio_file):
        try:
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                return None

            file_size = os.path.getsize(audio_file)
            if file_size == 0:
                print(f"Audio file is empty: {audio_file}")
                return None
            print(audio_file)
            transcription = self.whisper_model.transcribe(audio_file)
            return transcription["text"]
        
        except Exception as e:
            print(f"Error in transcribe audio: {str(e)}")
            return None

    def transcribe_all_audios(self, audio_files_dict):
        for url, audio_path in audio_files_dict.items():
            if not audio_path.endswith(".mp3"):
                print(f"Skipping non-mp3 file: {audio_path}")
                continue

            transciption = self.transcribe_audio(audio_path)

            if transciption is not None:
                # Add to transcription dictionary
                self.transcriptions_dict[url] = {
                    "url": url,
                    "audio_path": audio_path,
                    "transcription": transciption
                }
            else:
                print(f"Failed to transcribe audio: {audio_path}")
        
        return self.transcriptions_dict

### Download audio from video urls
downloader = YoutubeAudioDownloader(output_folder=r"./data/audios")
video_urls = ["https://www.youtube.com/watch?v=qP1JKWBBy80",
                "https://www.youtube.com/watch?v=_p58cZIHDG4"]
audio_files = downloader.download_multiple_audios(video_urls)
print("Downloaded audio file: ")
for audio_file in audio_files:
    print(audio_file)

### Transcribe audio from downloaded audios
# Load whisper model
whisper_model = whisper.load_model("medium", device=device)

# Initialize the AudioTranscriber
transcriber = AudioTranscriber(input_folder=r"./data/audios", whisper_model=whisper_model)

# Transcribe all audios in the input folder
transcriptions_dict = transcriber.transcribe_all_audios(audio_files)

for url, data in transcriptions_dict.items():
    print(f"URL: {data['url']}")
    print(f"Audio file: {data['audio_path']}")
    print(f"Transcription: {data['transcription'][:100]}")
    print("------")

# Write transcription to json file
audio_data = [
    {
        "url": value["url"],
        "audio_path": value["audio_path"],
        "transcription": value["transcription"]
    }
    for value in transcriptions_dict.values()
]

with open("./data/audios/audio_file.json", "w") as f:
    json.dump(audio_data, f, indent=2)

