# core/visualization.py
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Minimal Grad-CAM hooked on a named module (e.g. layer4).
    CAM is computed w.r.t. the top-1 predicted class."""
    def __init__(self, model, layer_name='layer4'):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x):
        """x: (1, 3, 32, 32) normalized tensor. Returns (32, 32) numpy heatmap
        in [0, 1] and the predicted class index."""
        self.model.zero_grad()
        out = self.model(x)
        pred_class = out.argmax(dim=1)
        score = out[0, pred_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, int(pred_class.item())


def to_input_tensor(img_uint8, device, mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)):
    """(H, W, 3) uint8 numpy array -> normalized (1, 3, H, W) tensor on device."""
    pil_like = img_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(pil_like).permute(2, 0, 1)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    t = (t - mean_t) / std_t
    return t.unsqueeze(0).to(device).requires_grad_(False)
