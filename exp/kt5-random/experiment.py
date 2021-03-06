import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('senior-capstone-commonsense-reasoning')+1])
sys.path.insert(1, project_path)
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.dataset import CSQA2DatasetWithVisibleMatrixForT5
from src.trainer import run_experiment

if __name__ == '__main__':

    config = {'model': 't5-large', 'max_seq_length': 512, 'max_entities': 40}
    print(config)
    tokenizer = T5Tokenizer.from_pretrained(config['model'])
    train_dataset = CSQA2DatasetWithVisibleMatrixForT5(data_path="../../data/csqa2/train.json",
                                                  tokenizer=tokenizer,
                                                  knowledge_path="../../data/knowledge/conceptnet.csv")
    eval_dataset = CSQA2DatasetWithVisibleMatrixForT5(data_path="../../data/csqa2/dev.json",
                                                 tokenizer=tokenizer,
                                                 knowledge_path="../../data/knowledge/conceptnet.csv")
    model = T5ForConditionalGeneration.from_pretrained(config['model'])
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        data_collator=train_dataset.collate_fn,
        output_dir="log",
        learning_rate=1e-4
    )

