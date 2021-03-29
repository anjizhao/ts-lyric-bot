
import glob
from typing import List


from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split


def get_all_datapoints(
    filepath: str = 'data/lyrics/*/*',
    combine_paragraphs=False,
    separate_songs=False,
) -> List[str]:
    song_lines: List[str] = []
    for lyrics_filename in glob.glob(filepath):
        with open(lyrics_filename, 'r') as lyrics_file:
            text = lyrics_file.read().strip()
            paragraphs = text.split('\n\n')
            for p in paragraphs:
                lines = p.split('\n')
                lines = [
                    line for line in lines
                    if line and not line.startswith('[')
                ]
                if combine_paragraphs:
                    song_lines.append(' '.join(lines))
                else:
                    song_lines.extend(lines)
            if separate_songs:
                song_lines.append('--')
    sent_tokenized = [
        sent for line in song_lines for sent in sent_tokenize(line)
    ]
    return sent_tokenized


def write_to_file(dataset: List[str], filename: str) -> None:
    lines = [line + '\n' for line in dataset]
    with open(filename, 'w') as outfile:
        outfile.writelines(lines)


if __name__ == '__main__':
    song_lines = get_all_datapoints()
    train, test = train_test_split(song_lines, random_state=2)
    write_to_file(train, 'data/lyrics_dataset_train.txt')
    write_to_file(test, 'data/lyrics_dataset_test.txt')
