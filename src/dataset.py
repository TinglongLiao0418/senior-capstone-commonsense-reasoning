import json
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class CSQA2DatasetBase(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_length = tokenizer.model_max_length
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
            self.max_seq_length
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


class CSQA2DatasetWithVisibleMatrix(CSQA2DatasetBase):
    def __init__(self, data_path, tokenizer, knowledge_path, max_entities=10, entity_sample='weighted'):
        super(CSQA2DatasetWithVisibleMatrix, self).__init__(data_path, tokenizer)
        self.max_entities = max_entities
        self.entity_sample = entity_sample
        self._create_lookup_table(knowledge_path)
    
    def _create_lookup_table(self, knowledge_path):
        self.knowledge = defaultdict(set)
        self.relations = {
            'antonym': 'is an antonym of',
            'atlocation': 'is located in',
            'capableof': 'is capable of',
            'causes': 'causes',
            'causesdesire': 'causes the desire to',
            'createdby': 'is created by',
            'definedas': 'is defined as',
            'derivedfrom': 'is derived from',
            'desires': 'desires',
            'distinctfrom': 'is distinct from',
            'entails': 'entails',
            'etymologicallyderivedfrom': 'is derived from',
            'etymologicallyrelatedto': 'has a common origin as',
            'formof': 'is a form of',
            'hasa': 'has a',
            'hascontext': 'is used in the context of',
            'hasfirstsubevent': 'is the first subevent of',
            'haslastsubevent': 'is the last subevent of',
            'hasprerequisite': 'has the prerequisite of',
            'hasproperty': 'has the property of',
            'hassubevent': 'has a subevent of',
            'instanceof': 'is an instance of',
            'isa': 'is a',
            'locatednear': 'is located near',
            'madeof': 'is made of',
            'mannerof': 'is a manner of',
            'motivatedbygoal': 'is motivated by',
            'notcapableof': 'is not capable of',
            'notdesires': 'not desires',
            'nothasproperty': 'not has the property of',
            'obstructedby': 'is obstructed by',
            'partof': 'is part of',
            'receivesaction': 'receives action of',
            'relatedto': 'is related to',
            'similarto': 'is similar to',
            'symbolof': 'symbolically represents',
            'synonym': 'is a synonym of',
            'usedfor': 'is used for',
            'dbpedia': ''
        }

        with open(knowledge_path, 'r', encoding='utf-8') as f:
            for line in f:
                relation, start, end = [i.lower().replace('_', ' ') for i in line.strip().split("\t")]
                try:
                    relation = self.relations[relation]
                except:
                    relation = relation
                value = "{} {}".format(relation, end)
                self.knowledge[start].add(value)

    def collate_fn(self, batch):
        input_ids, position_ids, visible_matrices, labels = [], [], [], []

        for example in batch:
            visible_matrix = np.zeros((self.max_seq_length, self.max_seq_length), dtype=int)
            position_id = np.zeros(self.max_seq_length, dtype=int)

            tokenized_prompt = self.tokenizer.tokenize(example['topic_prompt'])
            entities = self.knowledge[example['topic_prompt']]

            if len(entities) > self.max_entities:

                try:
                    if self.entity_sample != 'weighted':
                        raise ValueError('entity sample config not set correctly')

                    entities = list(entities)
                    cum_weights = [0 for _ in entities]

                    for i, e in enumerate(entities):
                        tokenized_entity = self.tokenizer.tokenize(e)
                        cum_weights[i] = sum([example['tokenized_question'].count(tokenized_e) for tokenized_e in tokenized_entity])

                    if sum(cum_weights) > 0:
                        cum_weights = [w + 1 for w in cum_weights]
                        selected_entities = set()

                        while (len(selected_entities) < self.max_entities):
                            idx = random.choices(range(len(entities)), cum_weights=cum_weights)[0]
                            selected_entities.add(entities[idx])
                            del entities[idx]
                            del cum_weights[idx]
                    
                        entities = selected_entities

                    else: 
                        entities = random.sample(entities, self.max_entities)

                except:
                    entities = random.sample(entities, self.max_entities)

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
                        len(input_tokens) + len(tokenized_entity) + remaining_len >= self.max_seq_length
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
                for j in range(curr_start, len(input_tokens)):
                    visible_matrix[i][j] = 1
                    visible_matrix[j][i] = 1
                    for k in range(curr_start, len(input_tokens)):
                        visible_matrix[j][k] = 1
                    
                    position_id[j] = entity_start + j - curr_start
            
            input_id = self.tokenizer.convert_tokens_to_ids(input_tokens)
            input_id += [self.tokenizer.pad_token_id for _ in range(self.max_seq_length - len(input_id))]

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


class CSQA2DatasetWithVisibleMatrixForT5(CSQA2DatasetWithVisibleMatrix):
    def __init__(self, data_path, tokenizer, knowledge_path, max_entities=10, entity_sample='weighted'):
        super(CSQA2DatasetWithVisibleMatrixForT5, self).__init__(data_path, tokenizer, knowledge_path, max_entities, entity_sample)


    def collate_fn(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for example in batch:
            question = "Question: " + example['question']
            input_id = self.tokenizer(question, return_tensors='pt').input_ids.squeeze()
            attention_mask = self.tokenizer(question, return_tensors='pt').attention_mask.squeeze()
            # expand the attention to 2D
            expanded_attention_mask = attention_mask.unsqueeze(0).repeat(attention_mask.size(0), 1)

            # find the prompt span
            prompt_start, prompt_end = -1, -1
            tokenized_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(example['topic_prompt']))
            i, j = 0, 0
            while i < input_id.size(0):

                if input_id[i].item() != tokenized_prompt[j]:
                    i += 1
                    j = 0
                else:
                    if j == len(tokenized_prompt) - 1:
                        prompt_start, prompt_end = j - len(tokenized_prompt) + 1, j + 1
                        break
                    else:
                        i += 1
                        j += 1

            if prompt_start == -1:
                entities = []
            else:
                num_entity = min(self.max_entities, len(self.knowledge[example['topic_prompt']]))
                entities = random.sample(self.knowledge[example['topic_prompt']], num_entity)

            for e in entities:
                tokenized_entity = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(e)))
                if tokenized_entity.size(0) + input_id.size(0) > self.max_seq_length - 5:  # Make room for decoder
                    break

                new_attention_block_upper_right = torch.zeros(input_id.size(0), tokenized_entity.size(0))
                new_attention_block_upper_right[:, prompt_start: prompt_end] = 1
                new_attention_block_lower_left = torch.permute(new_attention_block_upper_right, (1, 0))
                new_attention_block_lower_right = torch.ones(tokenized_entity.size(0), tokenized_entity.size(0))
                new_attention_block_lower = torch.cat((new_attention_block_lower_left, new_attention_block_lower_right), 1)
                expanded_attention_mask = torch.cat((expanded_attention_mask, new_attention_block_upper_right), 1)
                expanded_attention_mask = torch.cat((expanded_attention_mask, new_attention_block_lower), 0)

                input_id = torch.cat((input_id, tokenized_entity))

            padded_attention_mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.long)
            padded_attention_mask[:expanded_attention_mask.size(0), :expanded_attention_mask.size(0)] = expanded_attention_mask
            padded_input_ids = torch.ones(self.max_seq_length, dtype=torch.long) * self.tokenizer.pad_token_id
            padded_input_ids[:input_id.size(0)] = input_id

            label = torch.LongTensor(self.tokenizer(example['answer']).input_ids)

            input_ids.append(padded_input_ids)
            labels.append(label)
            attention_masks.append(padded_attention_mask)

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels)
        }


class CSQA2DatasetForT5(CSQA2DatasetWithVisibleMatrix):
    def __init__(self, data_path, tokenizer, knowledge_path, max_entities=10, entity_sample='weighted'):
        super(CSQA2DatasetForT5, self).__init__(data_path, tokenizer, knowledge_path, max_entities, entity_sample)


    def collate_fn(self, batch):

        for example in batch:
            example["statement"] = example['question']

            all_related_entities = self.knowledge[example['topic_prompt']]
            entities = random.sample(all_related_entities, min(self.max_entities, len(all_related_entities)))

            context = '. '.join([example['topic_prompt'] + ' ' + e for e in entities])
            example["context"] = context

        encoding = self.tokenizer(
            ["statement: " + example['statement'] + "context: " + example['context'] for example in batch],
            padding="longest",
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(
            [example["answer"] for example in batch],
            padding="longest",
            return_tensors="pt"
        )
        labels = target_encoding.input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class CorruptedConceptNet(Dataset):
    def __init__(self, path, tokenizer, max_context_num=10, corrupt_ratio=0.5):
        self.relations = {
            'antonym': 'is an antonym of',
            'atlocation': 'is located in',
            'capableof': 'is capable of',
            'causes': 'causes',
            'causesdesire': 'causes the desire to',
            'createdby': 'is created by',
            'definedas': 'is defined as',
            'derivedfrom': 'is derived from',
            'desires': 'desires',
            'distinctfrom': 'is distinct from',
            'entails': 'entails',
            'etymologicallyderivedfrom': 'is derived from',
            'etymologicallyrelatedto': 'has a common origin as',
            'formof': 'is a form of',
            'hasa': 'has a',
            'hascontext': 'is used in the context of',
            'hasfirstsubevent': 'is the first subevent of',
            'haslastsubevent': 'is the last subevent of',
            'hasprerequisite': 'has the prerequisite of',
            'hasproperty': 'has the property of',
            'hassubevent': 'has a subevent of',
            'instanceof': 'is an instance of',
            'isa': 'is a',
            'locatednear': 'is located near',
            'madeof': 'is made of',
            'mannerof': 'is a manner of',
            'motivatedbygoal': 'is motivated by',
            'notcapableof': 'is not capable of',
            'notdesires': 'not desires',
            'nothasproperty': 'not has the property of',
            'obstructedby': 'is obstructed by',
            'partof': 'is part of',
            'receivesaction': 'receives action of',
            'relatedto': 'is related to',
            'similarto': 'is similar to',
            'symbolof': 'symbolically represents',
            'synonym': 'is a synonym of',
            'usedfor': 'is used for',
            'dbpedia': ''
        }
        self._read_conceptnet(path)
        self.tokenizer = tokenizer
        self.max_context_num = max_context_num
        self.corrupt_ratio = corrupt_ratio

        self._corrupt_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def _read_conceptnet(self, path):
        self.data = []
        self.entity_pool = set()
        self.relation_pool = set()
        self.knowledge = defaultdict(lambda : defaultdict(set))
        self.entity_to_all_relations = defaultdict(list)

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                relation, start, end = [i.lower().replace('_', ' ') for i in line.strip().split("\t")]
                if relation in self.relations:
                    relation = self.relations[relation]

                example = {
                    'start': start,
                    'relation': relation,
                    'end': end
                }
                self.data.append(example)

                self.entity_pool.add(start)
                self.entity_pool.add(end)
                self.relation_pool.add(relation)
                self.knowledge[start][relation].add(end)
                self.entity_to_all_relations['start'].append(' '.join([start, relation, end]))

        self.entity_pool = list(self.entity_pool)
        self.relation_pool = list(self.relation_pool)

    def _corrupt_data(self):
        for example in self.data:

            if random.random() < self.corrupt_ratio:
                if random.random() < 0.5:
                    while True:
                        e = random.choice(self.entity_pool)
                        if e not in self.knowledge[example['start']][example['relation']]:
                            example['end'] = e
                            break
                else:
                    while True:
                        r = random.choice(self.relation_pool)
                        if example['end'] not in self.knowledge[example['start']][r]:
                            example['r'] = r
                            break
                example['answer'] = "no"
            else:
                example['answer'] = "yes"

    def collate_fn(self, batch):
        for example in batch:
            example["statement"] = ' '.join([example["start"], example["relation"], example["end"]]) + "."

            context = []
            choices = self.entity_to_all_relations['example_start']
            random.shuffle(choices)
            i = 0
            for c in choices:
                if i > self.max_context_num:
                    break
                if c == example["statement"]:
                    continue
                else:
                    context.append(c + ".")
            context = ' '.join(context)

            example["context"] = context

        encoding = self.tokenizer(
            ["statement: " + example['statement'] + "context: " + example['context'] for example in batch],
            padding="longest",
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(
            [example["answer"] for example in batch],
            padding="longest",
            return_tensors="pt"
        )
        labels = target_encoding.input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    datapath = '../data/csqa2/train.json'
    knowledgepath = '../data/knowledge/conceptnet.csv'
    dataset = CSQA2DatasetForT5(tokenizer=tokenizer, data_path=datapath, knowledge_path=knowledgepath, max_entities=2)
    # dataset = CorruptedConceptNet(tokenizer=tokenizer, path=knowledgepath)
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, batch_size=4)

    for i in tqdm(dataloader):
        pass