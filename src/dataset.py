from collections import defaultdict
import json
import pandas as pd
import numpy as np
import random
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
                example['tokenized_question'] = tokenizer.tokenize(example['question'])

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
            attention_mask = [1 for i in range(len(input_id))] + [self.tokenizer.pad_token_id for i in range(padding_length - len(input_id))]
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

class CSQA2DatasetWithVisibleMatrix(CSQA2DatasetBase):
    def __init__(self, config, data_path, tokenizer, knowledge_path):
        super(CSQA2DatasetWithVisibleMatrix, self).__init__(config, data_path, tokenizer)
        self._create_lookup_table(knowledge_path)
    
    def _create_lookup_table(self, knowledge_path):
        self.knowledge = defaultdict(lambda: set)

        with open(knowledge_path, 'r', encoding='utf-8') as f:
            for line in f:
                relation, start, end = [i.lower().replace('_', ' ') for i in line.strip().split("\t")]
                value = "{} {}".format(relation, end)
                self.knowledge[start].add(value)

    def collate_fn(self, batch):
        visible_matrix = np.zeros((self.config['max_seq_length'], self.config['max_seq_length']))
        position_ids = np.zeros(self.config['max_seq_length'])
        labels = []
        for example in batch:

            topic_prompt = example['topic_prompt']
            entities = self.knowledge[topic_prompt]

            # TODO: find the start and end of the topic prompt
            pos_start, pos_end = example['tokenized_question'].index(topic_prompt)

            if len(entities) > self.config['max_entities']:
                entities = random.sample(entities, self.config['max_entities'])
            
            abs_idx = pos_end + 1
            input_tokens = example['tokenized_question'][:abs_idx]
            for i in range(abs_idx):
                position_ids = i
                
            for e in entities:
                offset = 0
                for token in self.tokenizer.tokenize(e):
                    position_ids[abs_idx + offset] = pos_start + offset
                    input_tokens.extend(token)
            
            labels.append(example['label'])

        return {
            "input_ids": torch.LongTensor(input_ids),
            "position_ids": torch.LongTensor(position_ids),
            "visible_matrix": torch.LongTensor(visible_matrix),
            "labels": torch.LongTensor(labels)
        }


if __name__ == '__main__':
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    datapath = '../data/csqa2/dev.json'
    knowledgepath = '../data/knowledge/conceptnet.csv'
    dataset = CSQA2DatasetBase(config=config, tokenizer=tokenizer, data_path=datapath)

    dataset = CSQA2DatasetWithLink(config=config, tokenizer=tokenizer, data_path=datapath, knowledge_path=knowledgepath)
    print(dataset[0])
