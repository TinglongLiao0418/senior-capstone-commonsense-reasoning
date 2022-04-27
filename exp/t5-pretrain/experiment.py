import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('senior-capstone-commonsense-reasoning')+1])
sys.path.insert(1, project_path)
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.dataset import CorruptedConceptNet
from src.trainer import run_experiment

if __name__ == '__main__':

    config = {'model': 't5-large'}
    tokenizer = T5Tokenizer.from_pretrained(config['model'])
    train_dataset = CorruptedConceptNet(path="../../data/knowledge/conceptnet.csv", tokenizer=tokenizer)
    eval_dataset = CorruptedConceptNet(path="../../data/knowledge/conceptnet.csv", tokenizer=tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(config['model'])
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        epoch=1,
        evaluation_strategy="steps",
        eval_steps=5e4,
        save_strategy="steps",
        save_steps=5e4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        data_collator=train_dataset.collate_fn,
        output_dir="log/pretrain",
        learning_rate=1e-4
    )

