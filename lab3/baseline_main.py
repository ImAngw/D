from lab3.utils.utils import get_loaders, Configs, ClassificationContainer
from lab3.models.models import SimplerSentenceClassifier
import torch
from my_custom_ai.custom_train.train import CustomTraining



def main(configs):
    train_loader, val_loader = get_loaders(configs.model_name, configs.batch_size, return_loaders=True)
    container = ClassificationContainer(device=configs.device)
    model = SimplerSentenceClassifier(
        model_name=configs.model_name,
        hidden_dim=configs.hidden_dim,
        train_backbone=configs.train_backbone).to(configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    n_params = 0
    for param in model.parameters():
        n_params += param.flatten().shape[0]
    print(f'TOT Params: {n_params}')

    custom_training = CustomTraining(
        configs=configs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_on_validation=True,
        function_container=container,
    )

    custom_training.train()




if __name__ == '__main__':
    distil_bert = "distilbert/distilbert-base-uncased"
    sbert_mini = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_large = "sentence-transformers/all-mpnet-base-v2"

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