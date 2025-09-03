from utils.utils import Configs, return_loaders, ClassificationContainer, DistillationContainer
from models.models import MLP, CNN
from my_custom_ai.custom_train.train import CustomTraining
import torch




def reproduce_resnet_results(configs):
    train_loader, val_loader = return_loaders(configs.batch_size, configs.dataset)

    if configs.block_type == 'mlp':
        model = MLP(configs)
    else:
        model = CNN(
            dataset=configs.dataset,
            channels_hidden_conv=configs.channels_hidden_conv,
            img_reductions=configs.img_reductions,
            require_skip_connection=configs.require_skip_connection,
            depth=configs.depth
        )

    model = model.to(configs.device)

    n_params = 0
    for param in model.parameters():
        n_params += param.flatten().shape[0]
    print(f'TOT Params: {n_params}')

    container = ClassificationContainer(is_mlp=True if configs.block_type == 'mlp' else False, device=configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

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

def distillation_experiment(configs):
    train_loader, val_loader = return_loaders(configs.batch_size, configs.dataset, distillation=True)

    model = CNN(
        dataset=configs.dataset,
        channels_hidden_conv=configs.channels_hidden_conv,
        img_reductions=configs.img_reductions,
        require_skip_connection=configs.require_skip_connection,
        depth=configs.depth
    )

    model = model.to(configs.device)

    n_params = 0
    for param in model.parameters():
        n_params += param.flatten().shape[0]
    print(f'TOT Params: {n_params}')

    container = DistillationContainer(device=configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

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
    configs = Configs(
        checkpoint_dir='DLA_Labs/lab1/checkpoints',
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=118,

        # Train configs
        batch_size=64,
        num_epochs=10,

        # Optimizer configs
        lr=5e-5,
        dropout=0.01,

        # Early stopping configs
        require_early_stop=True,
        early_stopping_patience=10,
        early_stop_desc=True,

        # Dataset
        dataset='cifar10',

        # Model configs
        experiment_name='LargeCNN',
        depth=7,
        require_skip_connection=False,
        expansion_factor=1,
        block_type='cnn',

        # used when block_type == mlp
        dim_hidden_layers=[(28*28, 128)] + [(128, 128)] * 4,

        # used when block_type == cnn
        channels_hidden_conv=[(16, 32)] + [(32, 32)] * 6,
        img_reductions= [16] * 7,

        # Save on W&B
        save_on_wb=False,
        logger_update_at_each_epoch=True
    )

    reproduce_resnet_results(configs)
    # distillation_experiment(configs)
