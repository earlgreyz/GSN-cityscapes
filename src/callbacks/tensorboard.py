import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from .callback import Callback

class LoggerCallback(Callback):
    def __init__(self, logs_path):
        self.logs_path = logs_path

    def __call__(self, epoch, loss, accuracy, *args, **kwargs):
        with SummaryWriter(self.logs_path) as writer:
            writer.add_scalar('data/loss', loss, epoch)
            writer.add_scalar('data/accuracy', accuracy, epoch)


class VisualizerCallback(Callback):
    def __init__(self, logs_path):
        self.logs_path = logs_path

    def _apply_mask(self, image, mask):
        mask = np.dstack((mask, mask, mask)) * np.array((0, 255, 0))
        mask = mask.astype(np.uint8)
        return image // 2 + mask // 2

    def __call__(self, epoch, last_batch, *args, **kwargs):
        inputs, targets, predicted = last_batch

        image = (inputs[0].data.squeeze().cpu().float().numpy() * 255).astype(np.uint8)
        target = targets[0].data.squeeze().cpu().numpy().astype(np.uint8)
        predicted = predicted[0].data.squeeze().cpu().numpy().astype(np.uint8)

        C, H, W = image.shape

        correct = (target == predicted).astype(np.uint8)
        correct = np.dstack((correct, correct, correct)) * np.array((0, 255, 0), dtype=np.uint8)
        correct = Image.fromarray(correct)

        image = Image.fromarray(np.transpose(image, (1, 2, 0)))
        blended = Image.blend(image, correct, .5)

        target = Image.fromarray(target)
        predicted = Image.fromarray(predicted)

        acc_res = np.zeros((H, W * 2, C), dtype=np.uint8)
        acc_res[0:H, 0:W, :] = np.array(image, dtype=np.uint8)
        acc_res[0:H, W:2*W, :] = np.array(blended, dtype=np.uint8)

        pred_res = np.zeros((H, W * 2), dtype=np.uint8)
        pred_res[0:H, 0:W] = np.array(target, dtype=np.uint8)
        pred_res[0:H, W:2 * W] = np.array(predicted, dtype=np.uint8)

        acc_res = np.transpose(acc_res, (2, 0, 1))
        pred_res = np.expand_dims(pred_res, axis=0)

        with SummaryWriter(self.logs_path) as writer:
            writer.add_image('Epoch_{}_Accuracy'.format(epoch), acc_res, epoch)
            writer.add_image('Epoch_{}_Prediction'.format(epoch), pred_res, epoch)
