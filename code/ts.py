
import csv
import os
import requests
import time
from typing import Any, Dict, List, NamedTuple, Optional, Set

from dotenv import load_dotenv
import tqdm


load_dotenv()


GENIUS_API_BASE = 'https://api.genius.com'
GENIUS_CLIENT_ACCESS_TOKEN = os.getenv('GENIUS_CLIENT_ACCESS_TOKEN')
GENIUS_ARTIST_ID = 1177

DATA_DIR = 'data'


# create request session with auth header
_session = requests.Session()
_session.headers.update(
    {'Authorization': 'Bearer {}'.format(GENIUS_CLIENT_ACCESS_TOKEN)}
)


class Song(NamedTuple):
    song_id: int
    album_id: Optional[int]
    artist_id: Optional[int]
    song_title: Optional[str]
    album_name: Optional[str]
    artist_name: Optional[str]


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
    res = _session.get(url, params=params, headers=headers)
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
        song_title=song_data.get('title'),
        album_id=album.get('id'),
        album_name=album.get('name'),
        artist_id=primary_artist.get('id'),
        artist_name=primary_artist.get('name'),
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
) -> None:
    assert len(songs) > 0
    file_path = '{}/{}_{}.csv'.format(directory, filename, int(time.time()))
    with open(file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=songs[0]._fields)
        writer.writeheader()
        for s in songs:
            writer.writerow(s._asdict())


def read_songs_from_csv(file_path: str) -> List[Song]:
    songs = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            songs.append(Song(**row))  # type: ignore
    return songs


if __name__ == '__main__':
    song_ids = get_all_song_ids(per_page=50)
    all_songs = load_songs(song_ids)
    write_songs_to_csv(all_songs, 'all_songs')
