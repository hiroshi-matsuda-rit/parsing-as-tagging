import os

splits = {'train': [(1,815), (1001,1136)], 'dev': [(886,931), (1148,1151)], 'test': [(816,885), (1137,1147)]}
data_dir = 'datasets/CTB/ctb5.1_507K'
data_prefix = 'chtb_'
data_suffix = '.fid.utf8.conllu'
word_split_ch = ''

splits_expand = {'train': [], 'dev': [], 'test': []}

for file_name in os.listdir(data_dir):
    if file_name.startswith(data_prefix) and file_name.endswith(data_suffix):
        try:
            #sec_id = int(file_name.lstrip(data_prefix).rstrip(data_suffix))
            sec_id = int(file_name.split('_')[1].split('.')[0])
        except:
            continue
        found = False
        for split, sp_range_list in splits.items():
            for sp_min, sp_max in sp_range_list:
                if sp_min <= sec_id <= sp_max:
                    found = True
                    splits_expand[split].append(os.path.join(data_dir, file_name))
                    break
            if found:
                break

for split, file_list in splits_expand.items():
    with open(os.path.join(data_dir, f'{data_prefix}{split}{data_suffix}'), 'w') as fw:
        sent_id = 0
        words = []
        buffer = []
        for file_path in file_list:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('# '):
                        continue
                    elif len(line) == 0:
                        fw.write('# sent_id = {}-s{}-{}\n'.format(split, str(sent_id), file_path.split('/')[-1].split('.')[0]))
                        fw.write('# text = {}\n'.format(word_split_ch.join(words)))
                        fw.write('\n'.join(buffer))
                        fw.write('\n\n')
                        sent_id += 1
                        words.clear()
                        buffer.clear()
                    else:
                        words.append(line.split('\t')[1])
                        buffer.append(line)