import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('senior-capstone-commonsense-reasoning')+1])
sys.path.insert(1, project_path)
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.dataset import CSQA2DatasetForT5
from src.trainer import run_experiment

if __name__ == '__main__':

    config = {'model': 'log/pretrain/checkpoint-50000'}

    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    train_dataset = CSQA2DatasetForT5(data_path="../../data/csqa2/train.json",
                                      tokenizer=tokenizer,
                                      knowledge_path="../../data/knowledge/conceptnet.csv")
    eval_dataset = CSQA2DatasetForT5(data_path="../../data/csqa2/dev.json",
                                      tokenizer=tokenizer,
                                      knowledge_path="../../data/knowledge/conceptnet.csv")
    model = T5ForConditionalGeneration.from_pretrained(config['model'])
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epoch=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        data_collator=train_dataset.collate_fn,
        output_dir="log/csqa2-t5",
        learning_rate=1e-4,
    )

