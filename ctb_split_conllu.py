import os
import sys
from pathlib import Path

splits = {
    'train': set(range(1, 815 + 1)) | set(range(1001, 1136 + 1)),
    'dev': set(range(886, 931 + 1)) | set(range(1148, 1151 + 1)),
    'test': set(range(816, 885 + 1)) | set(range(1137, 1147 + 1)),
}
data_prefix = 'chtb_'
data_suffix = '.fid.utf8.conllu'


def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'datasets/CTB/ctb5.1_507K')
    print('data_dir =', data_dir, sys.stderr)
    splits_expand = {'train': [], 'dev': [], 'test': []}

    for file_name in os.listdir(data_dir):
        if file_name.startswith(data_prefix) and file_name.endswith(data_suffix):
            try:
                #sec_id = int(file_name.lstrip(data_prefix).rstrip(data_suffix))
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
        with open(os.path.join(data_dir, f'{data_prefix}{split}{data_suffix}'), 'w') as fw:
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
