import json

import pandas as pd
import torch
from torch.utils.data import Dataset


class CSQA2DatasetBase(Dataset):
    def __init__(self, config, data_path, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                example = json.loads(line)
                example['label'] = 1 if example['question'] == "yes" else 0
                example['tokenized_question'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example['question']))

                self.data[i] = example


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        input_ids, attention_masks, labels = [], [], []
        padding_length = min(
            max([len(example['tokenized_question']) + 2 for example in batch]),
            self.config['max_seq_length']
        )


        for example in batch:
            input_id = [self.tokenizer.cls_token_id] + \
                       example['tokenized_question'][-(padding_length-2):] + [self.tokenizer.sep_token_id]
            attention_mask = [1 for i in range(len(input_id))] + [0 for i in range(padding_length - len(input_id))]
            input_id += [self.tokenizer.pad_token_id for i in range(padding_length - len(input_id))]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(example['label'])

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_masks),
            "labels": torch.LongTensor(labels)
        }


class CSQA2DatasetWithLink(CSQA2DatasetBase):
    def __init__(self, config, data_path, tokenizer, knowledge_path):
        super(CSQA2DatasetWithLink, self).__init__(config, data_path, tokenizer)
        self.knowledge = pd.read_csv(knowledge_path, sep='\t', header=None)
        self._link_topic_at_start()

    def _link_topic_at_start(self):
        r, s, e = 0, 1, 2
        for example in self.data.values():

            topic = example['topic_prompt']
            links = []
            for _, lnk in self.knowledge[self.knowledge[s] == topic].iterrows():
                links.append({
                    'relation': lnk[r],
                    'start': lnk[s],
                    'end': lnk[e]
                })
            example['links'] = links


if __name__ == '__main__':
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    datapath = '../data/csqa2/dev.json'
    knowledgepath = '../data/knowledge/conceptnet.csv'
    dataset = CSQA2DatasetBase(config=config, tokenizer=tokenizer, data_path=datapath)

    dataset = CSQA2DatasetWithLink(config=config, tokenizer=tokenizer, data_path=datapath, knowledge_path=knowledgepath)
    print(dataset[0])
