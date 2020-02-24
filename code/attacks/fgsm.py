class FGSM:

    def __init__(self, model, epsilon, loss_fn, clip_min, clip_max):
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.loss_fn = loss_fn

    def attack(self, images, targets, targeted):
        self.model.eval()

        images_ = images.clone().detach()
        images_.requires_grad_()

        logits = self.model(images_)
        self.model.zero_grad()
        loss = self.loss_fn(logits, targets, reduction='sum')
        loss.backward()

        if targeted:
            perturbed_images = images_ - self.epsilon * images_.grad.sign()
        else:
            perturbed_images = images_ + self.epsilon * images_.grad.sign()

        if (self.clip_min is not None) or (self.clip_max is not None):
            perturbed_images.clamp_(min=self.clip_min, max=self.clip_max)

        return perturbed_images.detach()
