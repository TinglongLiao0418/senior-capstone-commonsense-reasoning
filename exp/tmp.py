from transformers import AutoModel

from transformers import AutoTokenizer

from src.dataset import CSQA2DatasetBase
from src.trainer import run_experiment
# from src.model import KBERT

if __name__ == '__main__':
    config = {'max_seq_length':512}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CSQA2DatasetBase(config, "data/csqa2/train.json", tokenizer)
    eval_dataset = CSQA2DatasetBase(config, "data/csqa2/dev.json", tokenizer)
    model = AutoModel.from_pretrained('bert-base-uncased')
    run_experiment(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CSQA2DatasetBase.collate_fn,
        output_dir=".",
    )