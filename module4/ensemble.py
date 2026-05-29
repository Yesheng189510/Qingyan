"""5-fold ensemble inference for acne severity grading.

Averages softmax probabilities across multiple fold models for more robust
predictions.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))

from module3_kd.student_model import StudentResNet18
from module3_kd.config_kd import CFG


class EnsembleGrader:
    """Ensemble of K-fold student models.  Averages softmax probabilities.

    Usage:
        checkpoints = [
            'output/fold_0/fold_0_best.pth',
            'output/fold_1/fold_1_best.pth',
            ...
        ]
        grader = EnsembleGrader(checkpoints)
        grade, probs = grader.predict('image.jpg')
    """

    def __init__(self, checkpoint_paths: list, device: str = 'cpu'):
        self.device = torch.device(device)
        cfg = dict(CFG)
        self.num_classes = cfg['num_classes']
        self.num_folds = len(checkpoint_paths)

        self.transform = transforms.Compose([
            transforms.Resize(cfg['resize_test']),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
        ])

        self.models = []
        for ckpt_path in checkpoint_paths:
            model = StudentResNet18(num_classes=self.num_classes)
            ckpt = torch.load(ckpt_path, map_location=self.device,
                              weights_only=True)
            model.load_state_dict(ckpt)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        print(f'Loaded {self.num_folds} models for ensemble.')

    @torch.no_grad()
    def predict(self, image_path: str):
        """Return (grade, avg_prob_dict, fold_prob_dicts)."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        all_probs = []
        for model in self.models:
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            all_probs.append(probs)

        stacked = torch.stack(all_probs, dim=0)         # (K, C)
        avg_probs = stacked.mean(dim=0)                  # (C,)
        grade = int(torch.argmax(avg_probs).item())

        prob_dict = {
            f'grade_{i}': round(float(avg_probs[i]), 4)
            for i in range(self.num_classes)
        }

        # Per-fold breakdown
        fold_details = {}
        for fi, probs in enumerate(all_probs):
            fold_details[f'fold_{fi}'] = {
                f'grade_{i}': round(float(probs[i]), 4)
                for i in range(self.num_classes)
            }

        return grade, prob_dict, fold_details

    @torch.no_grad()
    def predict_batch(self, image_paths: list):
        """Return list of (path, grade, prob_dict) tuples."""
        results = []
        for path in image_paths:
            grade, probs, _ = self.predict(path)
            results.append((path, grade, probs))
        return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble acne grading')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='List of fold checkpoint paths')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    grader = EnsembleGrader(args.checkpoints, device=args.device)
    grade, probs, folds = grader.predict(args.image)
    print(f'Image: {args.image}')
    print(f'Ensemble grade: {grade}')
    print(f'Average probabilities: {probs}')
    print(f'Per-fold details:')
    for fold, p in folds.items():
        print(f'  {fold}: {p}')
