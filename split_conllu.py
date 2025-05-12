import glob
import os
import sys
from pathlib import Path

dataset_splits = {
    "ptb": {
        'train': set(range(2, 21 + 1)),
        'dev': set(range(22, 22 + 1)),
        'test': set(range(23, 23 + 1))
    },
    "ctb": {
        'train': set(range(1, 815 + 1)) | set(range(1001, 1136 + 1)),
        'dev': set(range(886, 931 + 1)) | set(range(1148, 1151 + 1)),
        'test': set(range(816, 885 + 1)) | set(range(1137, 1147 + 1)),
    },
}
default_dir = {
    'ptb': 'datasets/PTB/treebank_3/parsed/mrg/wsj/*/*',
    'ctb': 'datasets/CTB/ctb5.1_507K/*',
}
data_suffix = '.fid.utf8.conllu'
output_filename_format = '{}-ud-{}.conllu'


def main():
    dataset = sys.argv[1]
    splits = dataset_splits[dataset]
    data_dir = Path(sys.argv[2] if len(sys.argv) > 2 else default_dir[dataset])
    print('data_dir =', data_dir, sys.stderr)
    splits_expand = {'train': [], 'dev': [], 'test': []}

    for file_name in glob.glob(data_dir):
        if file_name.endswith(data_suffix):
            try:
                sec_id = int(file_name.split('_')[1].split('.')[0])
            except:
                print('unrelated:', file_name, sys.stderr)
                continue
            for split, sp_range in splits.items():
                if sec_id in sp_range:
                    splits_expand[split].append(data_dir / file_name)
                    print(f'{split}:', file_name, sys.stderr)
                    break
            else:
                print('skipping:', file_name, sys.stderr)

    for split, file_list in splits_expand.items():
        with open(os.path.join(data_dir, output_filename_format.format(dataset, split)), 'w') as fw:
            for file_path in file_list:
                sent_id = 1
                words = []
                buffer = []
                with open(file_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('# '):
                            if not line.startswith("# text = ") or line.startswith("# sent_id = "):
                                print('skipping line:', line, sys.stderr)
                            continue
                        elif len(line) == 0:
                            text = ''.join(words).rstrip()
                            print(f'# sent_id = {split}-{file_path.stem()}-s{sent_id}', file=fw)
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
