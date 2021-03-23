
import csv
import os
import requests
import time
from typing import Any, Dict, List, NamedTuple, Optional, Set

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import tqdm
from unidecode import unidecode


load_dotenv()


GENIUS_API_BASE = 'https://api.genius.com'
GENIUS_CLIENT_ACCESS_TOKEN = os.getenv('GENIUS_CLIENT_ACCESS_TOKEN')
GENIUS_ARTIST_ID = 1177

DATA_DIR = 'data'

ALLOWED_ALBUM_IDS = {
    10982, 12682, 12731, 39005, 39094, 87979, 110728, 152556, 313068, 335029,
    350177, 350247, 361954, 463904, 520929, 545561, 597875, 597883, 621286,
    628059, 659925, 659926, 710140, 726425,
}

OVERRIDE_INCLUDE_SONG_IDS = {
    187017, 5077615, 132082, 187250, 132098, 5191847, 132079, 132092, 642957,
    186908, 3646550, 2887929, 186861,
}

OVERRIDE_EXCLUDE_SONG_IDS = {
    5077615, 5114093, 3331438, 187340, 1953921, 1983447, 186824, 186846,
    186852, 186870, 4969382
}

TS_ALBUM_NAMES = [
    'The Taylor Swift Holiday Collection',  # check this before 'Taylor Swift'
    'Taylor Swift',
    'Fearless',
    'Speak Now',
    'Red',
    '1989',
    'reputation',
    'Lover',
    'folklore',
    'evermore',
]


# create request session with auth header for genius api
_api_session = requests.Session()
_api_session.headers.update(
    {'Authorization': 'Bearer {}'.format(GENIUS_CLIENT_ACCESS_TOKEN)}
)

# session for scraping the actual lyric pages
_web_session = requests.Session()


class Song(NamedTuple):
    song_id: int
    album_id: Optional[int]
    artist_id: Optional[int]
    song_title: Optional[str]
    album_name: Optional[str]
    artist_name: Optional[str]
    url: str


def genius_get(
    path: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
) -> Dict[str, Any]:
    '''
    make get request to genius api at specified path.
    uses the module-level session, which includes the access token.
    '''
    if params is None:
        params = {}
    if headers is None:
        headers = {}
    url = GENIUS_API_BASE + path
    res = _api_session.get(url, params=params, headers=headers)
    res.raise_for_status()
    return res.json().get('response')


def get_all_song_ids(per_page: int = 20) -> List[int]:
    song_ids: Set[int] = set()
    next_page = 1
    while next_page is not None:
        params = {
            'page': next_page,
            'per_page': per_page,
        }
        data = genius_get(
            '/artists/{}/songs'.format(GENIUS_ARTIST_ID),
            params=params,
        )
        songs = data['songs']
        next_page = data['next_page']
        song_ids.update(song.get('id') for song in songs)
    return sorted(song_ids)


def _safe_clean_str(input_: Optional[str]) -> str:
    ''' convert input to a "clean" str (unidecoded, stripped). '''
    if input_ is None:
        return ''
    return unidecode(input_.replace('\u200b', '')).strip()


def get_song_data(song_id: int) -> Song:
    data: Dict[str, Dict] = genius_get(
        '/songs/{}'.format(song_id),
        params={'text_format': 'plain'}
    )
    song_data = data['song']
    album = song_data.get('album') or {}
    primary_artist = song_data.get('primary_artist') or {}
    return Song(
        song_id=song_id,
        song_title=_safe_clean_str(song_data.get('title')),
        album_id=album.get('id'),
        album_name=_safe_clean_str(album.get('name')),
        artist_id=primary_artist.get('id'),
        artist_name=_safe_clean_str(primary_artist.get('name')),
        url=song_data['url'],  # assuming this key always exists....
    )


def load_songs(song_ids: List[int]) -> List[Song]:
    songs: List[Song] = []
    for song_id in tqdm.tqdm(song_ids, desc='load_songs'):
        songs.append(get_song_data(song_id))
    return songs


def write_songs_to_csv(
    songs: List[Song],
    filename: str,
    directory: str = DATA_DIR,
) -> str:
    assert len(songs) > 0
    file_path = '{}/{}_{}.csv'.format(directory, filename, int(time.time()))
    with open(file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=songs[0]._fields)
        writer.writeheader()
        for s in songs:
            writer.writerow(s._asdict())
    return file_path


def read_songs_from_csv(file_path: str) -> List[Song]:
    songs = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                if v.isdigit() and k.endswith('_id'):
                    # convert ids back to numbers
                    row[k] = int(v)  # type: ignore
            songs.append(Song(**row))  # type: ignore
    return songs


def download_song_data() -> List[Song]:
    ''' fetch data from genius api, save data to file, return list of songs '''
    song_ids = get_all_song_ids(per_page=50)
    all_songs = load_songs(song_ids)
    return all_songs


def valid_ts(s: Song) -> bool:

    if s.song_id in OVERRIDE_INCLUDE_SONG_IDS:
        return True
    if s.song_id in OVERRIDE_EXCLUDE_SONG_IDS:
        return False

    if s.album_id not in ALLOWED_ALBUM_IDS:
        return False

    if not s.song_title:
        return False

    song_title = s.song_title.lower()
    if ('[' in song_title) and (']' in song_title):
        return False
    if 'remix' in song_title:
        return False
    if 'version' in song_title:
        return False
    if 'voice memo' in song_title:
        return False

    return True


def filter_ts_songs(all_songs: List[Song]) -> List[Song]:
    return [s for s in all_songs if valid_ts(s)]


def fetch_lyrics(url: str) -> str:
    res = _web_session.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    lyrics = soup.find_all('div', class_='lyrics')
    assert(len(lyrics) == 1)
    lyrics = lyrics[0]
    return lyrics.get_text()


def _standardize_album_name(album_name: Optional[str]) -> str:
    if album_name:
        if album_name == 'Taylor Swift (Best Buy Exclusive)':
            return 'Other'  # I Heart ?
        for known_name in TS_ALBUM_NAMES:
            if known_name in album_name:
                return known_name
        if album_name == '2004-2005 Demo CD':
            return 'Taylor Swift'  # a couple of songs are wrongly labeled...
        if album_name == 'Stripped: Raw & Real':
            return 'Fearless'  # Untouchable is wrongly labeled
        if album_name == '2003 Demo CD':
            return 'The Taylor Swift Holiday Collection'

    return 'Other'


def download_lyrics(song: Song) -> None:
    ''' scrape song url for lyrics and write them to a text file. '''
    album = _standardize_album_name(song.album_name)
    directory_name = '{}/lyrics/{}'.format(DATA_DIR, album)
    os.makedirs(directory_name, exist_ok=True)  # make sure directory exists
    file_path = '{}/{}.txt'.format(directory_name, song.song_title)
    lyrics = fetch_lyrics(song.url)
    with open(file_path, 'w') as txtfile:
        txtfile.write(_safe_clean_str(lyrics))


def download_all_lyrics(songs: List[Song]) -> None:
    for song in tqdm.tqdm(songs, desc='download_all_lyrics'):
        download_lyrics(song)
        time.sleep(1)  # bein extra conservative with the scraping :zany:


if __name__ == '__main__':
    # all_songs = download_song_data()
    # filename_all = write_songs_to_csv(all_songs, 'all_songs')
    filename_all = 'data/all_songs_1615501842.csv'
    all_songs = read_songs_from_csv(filename_all)

    # ts_songs = filter_ts_songs(all_songs)
    # filename_ts = write_songs_to_csv(ts_songs, 'ts_songs')
    filename_ts = 'data/ts_songs_1615501842.csv'
    ts_songs = read_songs_from_csv(filename_ts)

    download_all_lyrics(ts_songs)
    pass
