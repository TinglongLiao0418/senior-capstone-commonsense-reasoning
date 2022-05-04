from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments


def compute_metric_for_t5(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions[1].argmax(-1)

    r = 0
    for i in labels.size(0):
        if labels[i][0].item() == preds[i][0].item():
            r += 1

    return {
        'accuracy': r / labels.size(0)
    }



def compute_metric(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }


def run_experiment(model, train_dataset, eval_dataset, data_collator, output_dir='log', learning_rate=5e-6,
                   per_device_train_batch_size=8, per_device_eval_batch_size=8, gradient_accumulation_steps=1,
                   evaluation_strategy='epoch', eval_steps=2e4, save_strategy='epoch', save_steps=2e4,
                   epoch=5, seed=42, resume_from_checkpoint=False):
    train_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epoch,
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metric_for_t5
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
