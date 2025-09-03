from lab4.utils.utils import GetModelAndLoaders
from lab4.utils.robustCLS_utils import RobustCLSConfigs, RobustCLSContainer
from my_custom_ai.custom_train.train import CustomTraining
import torch



def train_main(configs):
    model_n_loaders = GetModelAndLoaders(configs.batch_size, configs.device)

    train_loader, val_loader = model_n_loaders.get_id_loaders()
    model = model_n_loaders.get_best_model()
    model.to(configs.device)

    container = RobustCLSContainer(device=configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)


    custom_training = CustomTraining(
        configs=configs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_on_validation=True,
        function_container=container,

        args_b=(),
        kwargs_b={
            'model': model,
            'eps': 1/255,
            'n_steps': 2
        },

        args_a=(),
        kwargs_a={
            'eps': 1 / 255,
            'n_steps': 2
        },

    )

    custom_training.train()



if __name__ == '__main__':
    configs = RobustCLSConfigs(
        experiment_name='robustCLS-training-new',
        checkpoint_dir='checkpoints',
        batch_size=128,
        num_epochs=100,
        seed=104,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        require_early_stop=True,
        early_stopping_patience=10,
        early_stop_desc=False,
        lr=1e-3,
    )

    train_main(configs)