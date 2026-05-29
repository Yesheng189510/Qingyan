"""Single-image and batch inference for acne severity grading."""

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

# Grade name mapping
GRADE_NAMES = {
    0: 'Mild (Grade 0)',
    1: 'Moderate (Grade 1)',
    2: 'Severe (Grade 2)',
    3: 'Very Severe (Grade 3)',
}


class AcneGrader:
    """Load a trained student model and classify acne severity.

    Usage:
        grader = AcneGrader('path/to/fold_0_best.pth')
        grade, probs = grader.predict('path/to/image.jpg')
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)

        cfg = dict(CFG)
        self.num_classes = cfg['num_classes']
        self.transform = transforms.Compose([
            transforms.Resize(cfg['resize_test']),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
        ])

        self.model = StudentResNet18(num_classes=self.num_classes)
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str):
        """Return (grade: int, probabilities: dict)."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        grade = int(torch.argmax(probs).item())

        prob_dict = {
            f'grade_{i}': round(float(probs[i]), 4) for i in range(self.num_classes)
        }
        return grade, prob_dict

    @torch.no_grad()
    def predict_batch(self, image_paths: list):
        """Return list of (grade, prob_dict) tuples."""
        results = []
        for path in image_paths:
            grade, probs = self.predict(path)
            results.append((path, grade, probs))
        return results


def grade_to_label(grade: int) -> str:
    return GRADE_NAMES.get(grade, f'Unknown ({grade})')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Acne severity inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    grader = AcneGrader(args.checkpoint, device=args.device)
    grade, probs = grader.predict(args.image)
    print(f'Image: {args.image}')
    print(f'Grade: {grade} ({grade_to_label(grade)})')
    print(f'Probabilities: {probs}')
