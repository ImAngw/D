from lab4.utils.utils import GetModelAndLoaders, ImagesPrinter
import torch
from lab4.utils.adversarial_utils import single_img_attack



def main(configs):
    loaders_n_models_container = GetModelAndLoaders(
        batch_size=configs.batch_size,
        device=configs.device,
    )

    model = loaders_n_models_container.get_best_model().to(configs.device)
    _, test_loader = loaders_n_models_container.get_id_loaders()

    img_printer = ImagesPrinter(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        labels=loaders_n_models_container.cifar10_classes
    )

    img, label = loaders_n_models_container.get_random_img(test_loader)
    img = img.to(configs.device).requires_grad_(True)
    y_true = label.to(configs.device)


    model.eval()
    out = model(img)
    y_pred = torch.argmax(out, dim=-1)

    if y_pred == y_true and configs.target_label != y_true:
        print(f'Attack!  True Label {y_true.item()}')
        adv_img, adv_label = single_img_attack(
            model=model,
            img=img,
            label=label.cpu().item(),
            target_label=configs.target_label,
            eps= 1/255,
            device=configs.device
        )

    else:
        print('Model already struggles with the image or target label is the same of the real label!')
        return

    img_printer(img, y_true, adv_img, adv_label)





if __name__ == '__main__':
    from types import SimpleNamespace

    configs = SimpleNamespace(
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        target_label=2    # None for untargeted
    )

    main(configs)