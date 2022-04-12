import sys
project_path = '/'.join(sys.path[0].split('/')[:sys.path[0].split('/').index('senior-capstone-commonsense-reasoning')+1])
sys.path.insert(1, project_path)
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

from src.dataset import CSQA2DatasetWithVisibleMatrix
from src.trainer import run_experiment

if __name__ == '__main__':

    config = {'max_seq_length': 512, 'max_entities': 10}
    model_config = BertConfig(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CSQA2DatasetWithVisibleMatrix(config=config,
                                                  data_path="../../data/csqa2/train.json",
                                                  tokenizer=tokenizer,
                                                  knowledge_path="../../data/knowledge/conceptnet.csv")
    eval_dataset = CSQA2DatasetWithVisibleMatrix(config=config,
                                                  data_path="../../data/csqa2/dev.json",
                                                  tokenizer=tokenizer,
                                                  knowledge_path="../../data/knowledge/conceptnet.csv")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    run_experiment(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
        output_dir=".",
    )

