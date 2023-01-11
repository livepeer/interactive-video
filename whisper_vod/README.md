# whisper_vod
## Prerequisite
1. Install Nvidia drivers.
2. Install CUDA toolkit 10.2.
3. Install cudnn8.2.1.
4. Install ffmpeg

## Installation
1. ```conda create -n whisper_vod python=3.10```
2. ```conda activate whisper_vod```
3. ```sudo apt install portaudio19-dev python-pyaudio python3-pyaudio```
4. ```pip install -r requirements.txt```

## Run
### VOD HLS Subtitles
```
python vod_subtitles.py -i <input>
# e.g, python vod_subtitles.py -i https://lp-playback.com/hls/ef3611aj4f0v1h7h/index.m3u8
```
Sample output:
```
Subtitled version available at https://wg.livepeer.com/hls/ef3611aj4f0v1h7h/index.m3u8
```
### Audio/Video to SRT
```
python audio2srt.py -f filename
```

## Workflow
1. Download input stream locally to mp4.
2. Transcribe mp4.
3. Generate SRT file based on transcription results. Whisper internally segments the input audio based on their spectrum, and has that segments data in the transcription output. The vod_subtitles use this segment as baseline and if any segment has more than 20 words and duration is more than 10 seconds, it further segments it.
4. Generate vtt files and subtitle manifest file using the srt and mp4 as input.
5. Create a new master playlist file with subtitles track.
    - Download master playlist from input url.
    - Update the variant video playlists with full url and add subtitle path. 
    E.g: 
    ```
    #EXT-X-STREAM-INF:PROGRAM-ID=0,BANDWIDTH=3629563,RESOLUTION=1920x1200,NAME="0-1200p0"
    1200p0/index.m3u8
    ```
    ```
    #EXT-X-STREAM-INF:PROGRAM-ID=0,BANDWIDTH=3629563,RESOLUTION=1920x1200,NAME="0-1200p0", SUBTITLES="subs0"
    http://lp-playback.com/hls/ef3611aj4f0v1h7h/1200p0/index.m3u8
    ```
    - Add subtitles playlist
    ```
    #EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs0",LANGUAGE="en",NAME="English",AUTOSELECT=YES,DEFAULT=YES,URI="https://wg.livepeer.com/subs/ef3611aj4f0v1h7h/index_vtt.m3u8"
    ```


