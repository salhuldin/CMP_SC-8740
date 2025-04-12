from transformers import Trainer, TrainingArguments

def get_training_args():
    return TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        report_to="none"
    )

def run_trainer(model, tokenizer, train_dataset, eval_dataset, data_collator, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    return trainer
