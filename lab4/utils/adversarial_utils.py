import torch
import torch.nn as nn
import torch.nn.functional as F



def top_correct_predictions(model, x_adv, labels, n=5):
    output = model(x_adv)

    probs = F.softmax(output, dim=-1)
    conf, preds = torch.max(probs, dim=1)

    mask_correct = preds.eq(labels)
    correct_idx = mask_correct.nonzero(as_tuple=True)[0]

    if correct_idx.numel() == 0:
        return None, None

    correct_conf = conf[correct_idx]
    correct_preds = preds[correct_idx]
    correct_imgs = x_adv[correct_idx]

    sorted_conf, sorted_idx = torch.sort(correct_conf, descending=True)

    top_idx = sorted_idx[:n]
    top_imgs = correct_imgs[top_idx]
    top_labels = correct_preds[top_idx]

    return top_imgs, top_labels

def random_batch_images(x_batch, labels, n=5):
    num_samples = x_batch.size(0)
    n_samples = min(n, num_samples)

    rand_idx = torch.randperm(num_samples)[:n_samples]

    imgs = x_batch[rand_idx]
    lbls = labels[rand_idx]

    return imgs, lbls

def batch_adv_attack(model, batch_imgs, labels, eps, n_steps, n_adv_imgs, random_imgs=True):
    criterion = nn.CrossEntropyLoss()
    if random_imgs:
        top_imgs, top_labels = random_batch_images(batch_imgs, labels, n_adv_imgs)
    else:
        top_imgs, top_labels = top_correct_predictions(model, batch_imgs, labels,n_adv_imgs)

    if top_imgs is None:
        return None, None

    x_adv = top_imgs.clone().detach().requires_grad_(True)

    for i in range(n_steps):
        x_adv.retain_grad()

        with torch.enable_grad():
            output = model(x_adv)
            model.zero_grad()
            loss = criterion(output, top_labels)
            loss.backward()

            x_adv = x_adv + eps * torch.sign(x_adv.grad)  # untargeted attack


    return x_adv.detach(), top_labels

def single_img_attack(model, img, label, eps, target_label=None, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    targeted_attack = False if target_label is None else True
    n = 0

    x_adv = img.clone().requires_grad_(True)

    if target_label is None:
        target_label = torch.tensor([label]).to(device)
    else:
        target_label = torch.tensor([target_label]).to(device)

    while True:
        x_adv.retain_grad()
        output = model(x_adv)
        prediction = torch.argmax(output, dim=-1)

        model.zero_grad()

        loss = criterion(output, target_label)
        loss.backward()


        if targeted_attack:
            x_adv -=  eps * torch.sign(x_adv.grad)
        else:
            x_adv += eps * torch.sign(x_adv.grad)

        if targeted_attack and prediction == target_label:
            print(f'- New Prediction: {prediction.item()}   Budget: {int(255 * n * eps)} / 255')
            break
        if not targeted_attack and prediction != target_label:
            print(f'- New Prediction: {prediction.item()}   Budget: {int(255 * n * eps)} / 255')
            break

        n += 1

    return x_adv, prediction