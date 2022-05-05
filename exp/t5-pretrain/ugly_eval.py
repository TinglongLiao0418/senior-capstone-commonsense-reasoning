import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('senior-capstone-commonsense-reasoning')+1])
sys.path.insert(1, project_path)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.dataset import CSQA2DatasetForT5

if __name__ == '__main__':
    model_path = 'log/csqa2-t5/checkpoint-4632'
    datapath = '../data/csqa2/train.json'
    knowledgepath = '../data/knowledge/conceptnet.csv'

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    dataset = CSQA2DatasetForT5(tokenizer=tokenizer, data_path=datapath, knowledge_path=knowledgepath, max_entities=2)
    dataloader = DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, batch_size=4)

    tot, right = 0, 0
    for data in tqdm(dataloader):
        prediction = model(input_ids=data['input_ids'], attention_mask=data['attention_mask']).logits.argmax(-1)
        labels = data['labels']
        result = (prediction == labels).all(dim=1)
        tot += result.size(0)
        right = result.int().sum().item()

    print(right / tot)

