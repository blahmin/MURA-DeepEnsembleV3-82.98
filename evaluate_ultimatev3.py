import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ultimate_ensemblev3 import UltimateEnsembleModel

class MURADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.body_parts = []

        body_parts = [d for d in os.listdir(root_dir) if d.startswith('XR_')]
        print("\nDataset composition:")
        for body_part in body_parts:
            part_dir = os.path.join(root_dir, body_part)
            part_count = 0
            for root, _, files in os.walk(part_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(root, file))
                        label = 1 if 'positive' in root.lower() else 0
                        self.labels.append(label)
                        self.body_parts.append(body_part)
                        part_count += 1
            print(f"{body_part}: {part_count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        body_part = self.body_parts[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, body_part, img_path

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_dir = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\valid"
    val_dataset = MURADataset(val_dir, transform=transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,  
        shuffle=False, 
        num_workers=2,
        pin_memory=True 
    )
    
    print(f"Total validation samples: {len(val_dataset)}")

    model = UltimateEnsembleModel(num_classes=2, beta=0.5).to(device)  
    checkpoint_path = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\ultimate_ensemble_v3.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
        if 'val_accuracy' in checkpoint:
            print(f"Model was saved with validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    part_correct = {}
    part_total = {}
    part_preds = {}
    part_labels = {}
    misclassified = []
    gate_values = []

    print("\nStarting evaluation...")
    with torch.no_grad():
        for inputs, labels, body_parts, paths in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs, gate = model(inputs, body_parts[0])
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            gate_values.extend(gate.cpu().numpy().flatten())
            
            for i, (pred, label, part, path) in enumerate(zip(predicted, labels, body_parts, paths)):
                if part not in part_correct:
                    part_correct[part] = 0
                    part_total[part] = 0
                    part_preds[part] = []
                    part_labels[part] = []
                
                part_total[part] += 1
                if pred == label:
                    part_correct[part] += 1
                else:
                    misclassified.append((path, label.item(), pred.item(), part))
                
                part_preds[part].append(pred.cpu().item())
                part_labels[part].append(label.cpu().item())
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del inputs, labels, outputs, predicted
            torch.cuda.empty_cache()

    accuracy = 100 * correct / total
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    print("\nOverall Results:")
    print(f"Total samples evaluated: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    print(f"Average Gate Value: {sum(gate_values) / len(gate_values):.4f}")
    
    print("\nPer Body Part Results:")
    for part in sorted(part_total.keys()):
        part_acc = 100 * part_correct[part] / part_total[part]
        part_kappa = cohen_kappa_score(part_labels[part], part_preds[part])
        print(f"\n{part}:")
        print(f"Accuracy: {part_acc:.2f}%")
        print(f"Kappa Score: {part_kappa:.4f}")
        print(f"Total Images: {part_total[part]}")
        print(f"Correct Predictions: {part_correct[part]}")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(part_labels[part], part_preds[part])
        print(cm)
        print("\nClassification Report:")
        print(classification_report(part_labels[part], part_preds[part], 
                                 target_names=['Normal', 'Abnormal']))

    save_path = os.path.join(os.path.dirname(val_dir), 'evaluation_results.txt')
    print(f"\nSaving detailed results to {save_path}")
    with open(save_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"=================\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Cohen's Kappa Score: {kappa:.4f}\n")
        f.write(f"Average Gate Value: {sum(gate_values) / len(gate_values):.4f}\n\n")
        for part in sorted(part_total.keys()):
            f.write(f"\n{part}:\n")
            f.write(f"Accuracy: {100 * part_correct[part] / part_total[part]:.2f}%\n")
            f.write(f"Total: {part_total[part]}, Correct: {part_correct[part]}\n")

    print("\nSaving misclassified cases to 'misclassified.txt'")
    with open('misclassified.txt', 'w') as f:
        f.write("Image Path,True Label,Predicted Label,Body Part\n")
        for path, true_label, pred_label, part in misclassified:
            f.write(f"{path},{true_label},{pred_label},{part}\n")

if __name__ == "__main__":
    evaluate_model()