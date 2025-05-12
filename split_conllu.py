import glob
import sys
from pathlib import Path

dataset_splits = {
    'ptb': {
        'train': set(range(2 * 100, (21 + 1) * 100)),
        'dev': set(range(22 * 100, (22 + 1) * 100)),
        'test': set(range(23 * 100, (23 + 1) * 100))
    },
    'ctb': {
        'train': set(range(1, 815 + 1)) | set(range(1001, 1136 + 1)),
        'dev': set(range(886, 931 + 1)) | set(range(1148, 1151 + 1)),
        'test': set(range(816, 885 + 1)) | set(range(1137, 1147 + 1)),
    },
}
default_path = {
    'ptb': 'datasets/ptb/treebank_3/parsed/mrg/wsj/*/*.conllu',
    'ctb': 'datasets/ctb/ctb5.1_507K/*.conllu',
}
output_path_format = '{}-ud-{}.conllu'


def main():
    dataset = sys.argv[1]
    splits = dataset_splits[dataset]
    data_path = sys.argv[2] if len(sys.argv) > 2 else default_path[dataset]
    print('data_path =', data_path, file=sys.stderr)
    splits_expand = {'train': [], 'dev': [], 'test': []}

    for file_path in glob.glob(data_path):
        file_path = Path(file_path)
        try:
            sec_id = int(file_path.name.split('_')[-1].split('.')[0])
        except:
            print('unrelated:', file_path, file=sys.stderr)
            continue
        for split, sp_range in splits.items():
            if sec_id in sp_range:
                splits_expand[split].append(file_path)
                print(f'{split}:', file_path, file=sys.stderr)
                break
        else:
            print('skipping:', file_path, file=sys.stderr)

    for split, file_list in splits_expand.items():
        with open(output_path_format.format(dataset, split), 'w', encoding='utf8') as fw:
            for file_path in file_list:
                sent_id = 1
                words = []
                buffer = []
                with open(file_path, 'r', encoding='utf8') as f:
                    for line in f:
                        line = line.rstrip()
                        if line.startswith('# '):
                            if not line.startswith('# text = ') and not line.startswith('# sent_id = '):
                                print(line, file=fw)
                            continue
                        elif line == "":
                            text = ''.join(words).rstrip()
                            print(f'# sent_id = {split}-{file_path.stem}-s{sent_id}', file=fw)
                            print(f'# text = {text}', file=fw)
                            print(*buffer, sep='\n', file=fw)
                            print(file=fw)
                            sent_id += 1
                            words = []
                            buffer = []
                        else:
                            buffer.append(line)
                            r = line.split('\t')
                            word = r[1]
                            if 'SpaceAfter=No' not in r[9]:
                                word += ' '
                            words.append(word)


if __name__ == '__main__':
    main()
