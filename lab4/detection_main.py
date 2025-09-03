import torch
from lab4.utils.utils import GetModelAndLoaders, ComputeScores
from lab4.models.models import AutoEncoder



def detection_main(batch_size, device, score_function, best_model_path):

    model_n_loaders = GetModelAndLoaders(batch_size, device, best_model_path)
    model = model_n_loaders.get_best_model()
    ae_model = AutoEncoder(
        batch_size=batch_size,
        device=device
    ).to(device)

    ae_model.load_state_dict(torch.load("checkpoints/autoencoder.pth"))

    _, id_loader = model_n_loaders.get_id_loaders()
    ood_loader = model_n_loaders.get_ood_loader()
    fake_loader = model_n_loaders.get_fake_loader(n_examples=1000)


    compute_scores = ComputeScores(
        model=model,
        ae_model=ae_model,
        device=device,
        id_loader=id_loader,
        score_function=score_function,
    )

    compute_scores.plot_scores_curves(fake_loader, is_ae=True)
    compute_scores.plot_scores_curves(fake_loader, is_ae=False)

    compute_scores.plot_scores_curves(ood_loader, is_ae=True)
    compute_scores.plot_scores_curves(ood_loader, is_ae=False)


if __name__ == '__main__':
    detection_main(
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
        score_function='MAX_SOFTMAX',   # Allowed functions: MAX_LOGIT, MAX_SOFTMAX
        best_model_path="checkpoints/robustCLS.pth" # teacher path: "../lab1/checkpoints/teacher.pth", robustCLS: "checkpoints/robustCLS-training.pth"
    )
