from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments


def compute_metric(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }


def run_experiment(model, train_dataset, eval_dataset, data_collator, output_dir,
                   gradient_accumulation_steps=1, epoch=5, seed=42):
    train_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epoch,
        save_strategy='epoch',
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metric
    )

    trainer.train()
