from lab3.utils.utils import get_loaders, Configs
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model






def main(configs):
    train_loader, val_loader = get_loaders(configs.model_name, configs.batch_size, return_loaders=False)
    conf = AutoConfig.from_pretrained(
        configs.model_name,
        num_classes=2,
        hidden_dropout_prob=0.2,
        classifier_dropout=0.3
    )

    model = AutoModelForSequenceClassification.from_pretrained(configs.model_name, config=conf).to(configs.device)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lora_config = LoraConfig(
        r=16,
        target_modules=["q_lin", "k_lin", 'v_lin'],
        task_type=TaskType.SEQ_CLS,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
        }

    training_args = TrainingArguments(
        output_dir="Lab2/checkpoints",
        eval_strategy='epoch',
        save_strategy="epoch",

        per_device_train_batch_size=configs.batch_size,
        per_device_eval_batch_size=configs.batch_size,
        eval_steps=1,
        num_train_epochs=10,
        learning_rate=1e-4,
        dataloader_pin_memory=False,
        logging_steps= len(train_loader) // configs.batch_size,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == '__main__':
    distil_bert = "distilbert/distilbert-base-uncased"

    configs = Configs(
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir='../lab3/checkpoints',
        seed=104,

        # Training configs
        batch_size=64,
        num_epochs=10,
        lr=1e-4,
        require_early_stop=True,
        early_stopping_patience=10,
        train_backbone=False,

        # Experiment Configs
        model_name=distil_bert,
        # hidden_dim=384,
        experiment_name='DistilBert',
        save_on_wb=False
    )

    main(configs)