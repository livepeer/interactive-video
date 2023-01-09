from datetime import timedelta
import os
import whisper
import argparse

def transcribe_audio(path):
    model = whisper.load_model("base") # Change this to your desired model
    print("Whisper model loaded.")
    transcribe = model.transcribe(audio=path)
    segments = transcribe['segments']

    for segment in segments:
        startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
        text = segment['text']
        segmentId = segment['id']+1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

        srtFilename = path.split(".")[0] + ".srt"
        with open(srtFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(segment)

    return srtFilename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename',
                    help='-f <filename>.\n'
                         )
    args = parser.parse_args()
    filename = args.filename
    transcribe_audio(filename)