import ffmpeg
import whisper
from datetime import timedelta
import argparse
import subprocess
import requests
import m3u8
import os
import webvtt

model = whisper.load_model("base")

def generate_hls(vid_in, sub_in, out="test"):
    args_out = {
        "c": "copy",
        "f": "hls",
        "hls_playlist_type": "vod",
        "var_stream_map": "v:0,a:0,s:1",
        "master_pl_name": "master.m3u8",
    }

    # input_ffmpeg = ffmpeg.input(vid_in)
    # input_ffmpeg_sub = ffmpeg.input(sub_in)
    #
    # input_video = input_ffmpeg['v']
    # input_audio = input_ffmpeg['a']
    # input_subtitles = input_ffmpeg_sub['s']
    #
    # output_ffmpeg = ffmpeg.output(input_video, input_audio, input_subtitles,
    #                               f"hls/{out}/%v/index.m3u8", **args_out, scodec="webvtt")
    # print(ffmpeg.compile(output_ffmpeg))
    output_ffmpeg = f"ffmpeg -i whisperme.mp4 -i whisperme.srt -c:v copy -c:a copy -c:s webvtt " \
                    "-f hls -hls_playlist_type vod -var_stream_map v:0,a:0,s:0 -master_pl_name master.m3u8 " \
                    f"-hls_time 6 -hls_segment_filename {out}/seg-%d.ts {out}/index.m3u8"
    output_ffmpeg = output_ffmpeg.split()
    subprocess.run(output_ffmpeg)

    # master_pl = f"hls/{out}/master.m3u8"
    # with open(master_pl, "r") as f:
    #     contents = f.readlines()

    # contents[2] = contents[2].rstrip() + ", SUBTITLES=\"subs\" \n"
    # contents.insert(2, "#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID='subs',NAME='English',"
    #                    "DEFAULT=NO,FORCED=NO,URI='0/index_vtt.m3u8',LANGUAGE='en'\n")


    # with open(master_pl, "w") as f:
    #     contents = "".join(contents)
    #     f.write(contents)
    try:
        os.remove(vid_in)
        os.remove(sub_in)
    except OSError:
        pass

def update_hls_transcript(
        input,
        subtitle_path=None,
):
    # download hls to mp4
    print("downloading hls to mp4...")
    args_out = {
        "c": "copy",
    }
    ffmpeg.input(input).output("whisperme.mp4", **args_out).run()
    
    print("whisper is transcribing mp4")
    # transcribe input audio
    srtFilename = transcribe_audio("whisperme.mp4")
    
    
    input_url_split = input.split("/")
    # vtt = webvtt.from_srt(srtFilename)
    # vtt.save()
    if subtitle_path is None:
        subtitle_path = input_url_split[-2]
    
    
    # webvtt.segment("whisperme.vtt", "/var/www/html/subs/" + subtitle_path)
    # Create a new master playlist with updated variants and subtitles manifest
    res = requests.get(input)
    pl = res.text
    input_pl = m3u8.loads(pl)
    if input_pl.is_variant:
        input_url_prefix = "/".join(input_url_split[:-1])
        for pl in input_pl.playlists:
            pl.uri = input_url_prefix + "/" + pl.uri
            pl.stream_info.subtitles = 'subs0'
    
    # create vtt files from srt file
    generate_hls("whisperme.mp4", "whisperme.srt", f"/var/www/html/subs/{subtitle_path}")

    # add subtitles manifest to master playlist
    master_pl = "/var/www/html/hls/" + subtitle_path + "/index.m3u8"
    input_pl.dump(master_pl)

    with open(master_pl, "r") as f:
        contents = f.readlines()

    for i in range(0, len(contents)):
        if contents[i].startswith("#EXT-X-STREAM-INF"):
            contents[i] = contents[i].rstrip("\n") + ", SUBTITLES=\"subs0\"\n"

    subtitle_contents = "#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"subs0\",LANGUAGE=\"en\",NAME=\"English\"," \
                        "AUTOSELECT=YES,DEFAULT=YES,URI=\"https://wg.livepeer.com/subs/{}/index_vtt.m3u8\"\n".format(subtitle_path)
    contents.append(subtitle_contents)

    with open(master_pl, "w") as f:
        contents = "".join(contents)
        f.write(contents)

    print("Subtitled version available at https://wg.livepeer.com/hls/{}/index.m3u8".format(subtitle_path))



def transcribe_audio(path):
    print("Whisper model loaded.")
    transcribe = model.transcribe(audio=path)
    segments = transcribe['segments']

    segmentId = 1
    for segment in segments:
        text = segment['text']
        wordcount = len(text.split())
        segStart = int(segment['start'])
        segEnd = int(segment['end'])
        textList = text.split()

        if wordcount > 20 and (segEnd - segStart) > 10:
            numSubSegs = int(wordcount / 20) + 1
            subsegWordCnt = int(wordcount / numSubSegs)
            duration = (segEnd - segStart) / numSubSegs
            for i in range(0, numSubSegs):
                startTime = str(0) + str(timedelta(seconds=int(segStart + duration * i))) + ',000'
                endTime = str(0) + str(timedelta(seconds=int(segStart + duration * i + duration))) + ',000'

                if i == (numSubSegs - 1):
                    subList = textList[i * subsegWordCnt:]
                else:
                    subList = textList[i * subsegWordCnt : (i + 1) * subsegWordCnt]

                subtext = " ".join(subList)

                segment = f"{segmentId}\n{startTime} --> {endTime}\n{subtext[1:] if subtext[0] == ' ' else subtext}\n\n"

                srtFilename = path.split(".")[0] + ".srt"
                with open(srtFilename, 'a', encoding='utf-8') as srtFile:
                    srtFile.write(segment)
                segmentId += 1

        else:
            startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
            endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'


            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

            srtFilename = path.split(".")[0] + ".srt"
            with open(srtFilename, 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)
            segmentId += 1

    return srtFilename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                    help='-i <input hls>.\n'
                         )
    args = parser.parse_args()
    input = args.input
    update_hls_transcript(input)