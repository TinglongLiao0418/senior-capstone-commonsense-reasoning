import json
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class CSQA2DatasetBase(Dataset):
    def __init__(self, config, data_path, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                example = json.loads(line)
                example['label'] = 1 if example['answer'] == "yes" else 0
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
                       self.tokenizer.convert_tokens_to_ids(example['tokenized_question'][-(padding_length-2):]) +\
                       [self.tokenizer.sep_token_id]
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
        self.knowledge = defaultdict(set)

        with open(knowledge_path, 'r', encoding='utf-8') as f:
            for line in f:
                relation, start, end = [i.lower().replace('_', ' ') for i in line.strip().split("\t")]
                value = "{} {}".format(relation, end)
                self.knowledge[start].add(value)

    def collate_fn(self, batch):
        input_ids, position_ids, visible_matrices, labels = [], [], [], []

        for example in batch:
            visible_matrix = np.zeros((self.config['max_seq_length'], self.config['max_seq_length']), dtype=int)
            position_id = np.zeros(self.config['max_seq_length'], dtype=int)

            tokenized_prompt = self.tokenizer.tokenize(example['topic_prompt'])
            entities = self.knowledge[example['topic_prompt']]

            if len(entities) > self.config['max_entities']:
                entities = random.sample(entities, self.config['max_entities'])

            # find the start and end of the topic prompt in the question
            prompt_start, prompt_end = -1, -1
            for i in range(len(example['tokenized_question'])):
                if example['tokenized_question'][i] == tokenized_prompt[0]:
                    n = 1
                    while n < len(tokenized_prompt) and example['tokenized_question'][i + n] == tokenized_prompt[n]:
                        n += 1
                    
                    if n == len(tokenized_prompt):
                        prompt_start = i
                        prompt_end = i + n - 1
                        break
            
            input_tokens = [self.tokenizer.cls_token] + example['tokenized_question'][:prompt_end + 1]
            remaining_len = len(example['tokenized_question'][prompt_end + 1:])
            entity_start = prompt_end + 2
            for i in range(entity_start):
                for j in range(entity_start):
                    visible_matrix[i][j] = 1

                position_id[i] = i

            curr_start = entity_start
            for e in entities:
                tokenized_entity = self.tokenizer.tokenize(e)
                if (
                        prompt_end < 0 or 
                        len(input_tokens) + len(tokenized_entity) + remaining_len >= self.config['max_seq_length']
                    ):
                    # if not finding the prompt within the question
                    # or adding the entity makes the seq exceed max_seq_length
                    break

                for i in range(len(tokenized_entity)):
                    for j in range(prompt_start + 1, entity_start):
                        # words within the prompt and entity can see each other
                        visible_matrix[curr_start + i][j] = 1
                        visible_matrix[j][curr_start + i] = 1

                    for j in range(len(tokenized_entity)):
                        visible_matrix[curr_start + i][curr_start + j] = 1
                    position_id[curr_start + i] = entity_start + i   

                input_tokens.extend(tokenized_entity)
                curr_start += len(tokenized_entity)

            input_tokens += example['tokenized_question'][prompt_end + 1:] + [self.tokenizer.sep_token]
            for i in range(entity_start):
                for j in range(curr_start, len(input_tokens) + 1):
                    visible_matrix[i][j] = 1
                    visible_matrix[j][i] = 1
                    for k in range(curr_start, len(input_tokens) + 1):
                        visible_matrix[j][k] = 1
                    
                    position_id[j] = entity_start + j - curr_start
            
            input_id = self.tokenizer.convert_tokens_to_ids(input_tokens)
            input_id += [self.tokenizer.pad_token_id for _ in range(self.config['max_seq_length'] - len(input_id))]

            input_ids.append(input_id)
            position_ids.append(position_id)
            visible_matrices.append(visible_matrix)
            labels.append(example['label'])

        return {
            "input_ids": torch.LongTensor(input_ids),
            "position_ids": torch.LongTensor(position_ids),
            "attention_mask": torch.LongTensor(visible_matrices),
            "labels": torch.LongTensor(labels)
        }


if __name__ == '__main__':
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    datapath = '../data/csqa2/dev.json'
    knowledgepath = '../data/knowledge/conceptnet.csv'
    dataset = CSQA2DatasetWithVisibleMatrix(config=config, tokenizer=tokenizer, data_path=datapath, knowledge_path=knowledgepath)

    print(dataset[0])