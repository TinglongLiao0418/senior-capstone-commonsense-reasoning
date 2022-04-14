import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('senior-capstone-commonsense-reasoning')+1])
sys.path.insert(1, project_path)
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

from src.dataset import CSQA2DatasetWithVisibleMatrix
from src.trainer import run_experiment

if __name__ == '__main__':
    max_entities = 20
    print("MAX ENTITIES: " + max_entities)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CSQA2DatasetWithVisibleMatrix(data_path="../../data/csqa2/train.json",
                                                  tokenizer=tokenizer,
                                                  knowledge_path="../../data/knowledge/conceptnet.csv",
                                                  max_entities=max_entities,
                                                  entity_sample='weighted')
    eval_dataset = CSQA2DatasetWithVisibleMatrix(data_path="../../data/csqa2/dev.json",
                                                 tokenizer=tokenizer,
                                                 knowledge_path="../../data/knowledge/conceptnet.csv",
                                                 max_entities=max_entities,
                                                 entity_sample='weighted')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
        output_dir="log",
    )

