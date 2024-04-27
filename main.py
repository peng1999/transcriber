import argparse
from pathlib import Path
import re

import bs4
import requests
import whisper
from whisper.utils import WriteTXT
from yt_dlp import YoutubeDL


def raise_if_not_single(results, name):
    if len(results) == 0:
        raise Exception(f"{name} not found in RSS feed")
    if len(results) > 1:
        raise Exception(f"Multiple {name} found in RSS feed")


def parse_rss(url, title_text):
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.content, 'xml')

    titles = soup.find_all('title', string=re.compile(title_text))
    raise_if_not_single(titles, title_text)
    title: bs4.Tag = titles[0]

    enclosures = title.find_next_siblings('enclosure', type=re.compile('audio'))
    raise_if_not_single(enclosures, 'enclosure')
    enclosure = enclosures[0]
    assert isinstance(enclosure, bs4.Tag)
    url = enclosure.attrs['url']
    assert isinstance(url, str)

    return url


OUT_TMPL = "%(id)s.%(ext)s"

def download_audio(url):
    ydl_options = dict(
        format='m4a/bestaudio/best',
        postprocessors=[
            dict(key='FFmpegExtractAudio', preferredcodec='m4a')
        ],
        outtmpl=OUT_TMPL,
    )
    with YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=True)

    assert isinstance(info, dict)
    filename = OUT_TMPL % info
    return filename

def transcribe_audio(audio_path: str, output_dir: str, model):
    if not Path(output_dir).is_dir():
        raise Exception(f"Directory {output_dir} not found.")

    model = whisper.load_model("medium", download_root="/home/pgw/data/my/cache")

    result = model.transcribe(audio_path, verbose=True, language='zh')
    writer = WriteTXT(output_dir)
    writer(result, audio_path)

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using Whisper model')
    parser.add_argument('--url', required=True, help='URL of the video')
    parser.add_argument('--output-dir', default='.', help='Output file path')
    parser.add_argument('--model', default='medium', help='Whisper model')
    parser.add_argument('--rss-title', help='RSS title text')
    args = parser.parse_args()

    if args.rss_title:
        url = parse_rss(args.url, args.rss_title)
        resp = requests.get(url)
        audio_file = Path(url.split('/')[-1])
        audio_file.write_bytes(resp.content)
        print(f"Downloaded audio to {audio_file}")
    else:
        audio_file = download_audio(args.url)
    transcribe_audio(str(audio_file), args.output_dir, args.model)

if __name__ == '__main__':
    main()

