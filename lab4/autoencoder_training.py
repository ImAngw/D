from lab4.models.models import AutoEncoder
from lab4.utils.utils import GetModelAndLoaders
from lab4.utils.autoencoder_utils import AutoencoderConfigs, AutoencoderContainer
from my_custom_ai.custom_train.train import CustomTraining
import torch



def main(configs):
    model_n_loaders = GetModelAndLoaders(configs.batch_size, configs.device)
    train_loader, val_loader = model_n_loaders.get_id_loaders()

    model = AutoEncoder(
        batch_size=configs.batch_size,
        device=configs.device,
    ).to(configs.device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    container = AutoencoderContainer(device=configs.device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=configs.lr)

    custom_training = CustomTraining(
        configs=configs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_on_validation=True,
        function_container=container
    )

    custom_training.train()




if __name__ == '__main__':
    configs = AutoencoderConfigs(
        experiment_name='autoencoder-training',
        checkpoint_dir='checkpoints',
        batch_size=32,
        num_epochs=50,
        seed=104,
        device="cuda" if torch.cuda.is_available() else "cpu",
        require_early_stop=True,
        early_stopping_patience=10,
        early_stop_desc=False,
        lr=1e-4,
    )

    main(configs)
