
import glob
from typing import List


from sklearn.model_selection import train_test_split


def get_all_datapoints() -> List[str]:
    song_lines: List[str] = []
    for lyrics_filename in glob.glob('data/lyrics/*/*'):
        with open(lyrics_filename, 'r') as lyrics_file:
            text = lyrics_file.read().strip()
            lines = text.split('\n')
            for line in lines:
                if line and not line.startswith('['):
                    song_lines.append(line)
    return song_lines


def write_to_file(dataset: List[str], filename: str) -> None:
    lines = [line + '\n' for line in dataset]
    with open(filename, 'w') as outfile:
        outfile.writelines(lines)


if __name__ == '__main__':
    song_lines = get_all_datapoints()
    train, test = train_test_split(song_lines, random_state=2)
    write_to_file(train, 'data/lyrics_dataset_train.txt')
    write_to_file(test, 'data/lyrics_dataset_test.txt')
