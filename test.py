import whisper

whisper_model = whisper.load_model("base")
path = "./data/harvard.wav"
transciption = whisper_model.transcribe(path)
print(transciption)