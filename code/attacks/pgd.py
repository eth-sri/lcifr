import torch

from attacks import FGSM


class PGD:

    def __init__(self, model, epsilon, loss_fn, clip_min, clip_max):
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.loss_fn = loss_fn
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def attack(self, alpha, images, iterations, targets, targeted, num_restarts, random_start):
        self.model.eval()

        images_ = images.clone().detach()
        images_min, images_max = images_ - self.epsilon, images_ + self.epsilon

        fgsm = FGSM(self.model, alpha, self.loss_fn, self.clip_min, self.clip_max)

        if not random_start:
            num_restarts = 1

        for i in range(num_restarts):
            if random_start:
                images_ = images.clone().detach() + torch.mul(
                    self.epsilon,
                    torch.rand_like(images, device=self.device).uniform_(-1, 1)
                )

            for it in range(iterations):
                images_ = fgsm.attack(images_, targets, targeted)

                # project onto epsilon-ball around original images
                images_ = torch.max(images_min, images_)
                images_ = torch.min(images_max, images_)

        return images_.detach()
