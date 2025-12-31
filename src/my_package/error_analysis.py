import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve

from my_package.databases.affine_transformer import AffineTransformer
from my_package.databases.lfw2_dataset import LFW2Dataset
from my_package.learners.paper_learner14 import PaperLearner14
from my_package.constants import BASE_DIR
from my_package.pipeline import setup_loaders

def run_error_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = BASE_DIR / 'model_weights.pth'
    
    learner = PaperLearner14(device=device)
    learner.load_model(model_path)
    learner.setup_loss(0.5)
    learner.set_train(False)

    img_transformer = AffineTransformer()
    train_loader, val_loader, test_loader, train_pos_pct = setup_loaders(resize_size=learner.resize_size, use_foreground=learner.use_foreground, 
                                                          batch_size=32, normalize_imgs=learner.normalize_imgs, num_workers=4,
                                                          img_transformer=img_transformer, val_ratio=0.2)

    all_probs, all_labels, all_imgs = [], [], []
    
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1_dev, img2_dev = img1.to(device), img2.to(device)
            label = label.to(device, non_blocking=True, dtype=torch.float32).unsqueeze(1)

            probs, _ = learner.process_batch(img1_dev, img2_dev, label, is_train=False)
            
            all_probs.append(probs.cpu())
            all_labels.append(label.cpu())
            all_imgs.append(torch.stack([img1, img2], dim=1))

    probs = torch.cat(all_probs).numpy().ravel() # Flatten to 1D
    labels = torch.cat(all_labels).numpy().ravel() # Flatten to 1D
    imgs = torch.cat(all_imgs).numpy()

    # 1. Threshold and Boxplot Analysis
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot A: F1 Score vs Threshold
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(thresholds, f1_scores[:-1], color='blue', lw=2)
    ax0.axvline(best_threshold, color='red', linestyle='--')
    ax0.set_title(f'F1 Optimization (Best Thresh: {best_threshold:.2f})')
    ax0.set_ylabel('F1 Score')

    # Plot B: Probability Boxplots (Confidence Spread)
    ax1 = fig.add_subplot(gs[0, 1])
    data_to_plot = [probs[labels == 0], probs[labels == 1]]
    ax1.boxplot(data_to_plot, tick_labels=['Different (0)', 'Same (1)'], patch_artist=True)
    ax1.axhline(0.5, color='black', linestyle=':', alpha=0.5)
    ax1.set_title('Confidence Distribution per Label')
    ax1.set_ylabel('Predicted Probability')

    # 2. Most "Wrong" Examples (Top 5 per label)
    # Most wrong for Label 1 (Same): Lowest predicted probabilities
    # Most wrong for Label 0 (Diff): Highest predicted probabilities
    wrong_same_idx = np.argsort(probs[labels == 1])[:5]
    wrong_diff_idx = np.argsort(probs[labels == 0])[-5:][::-1]

    def plot_top_failures(indices, label_mask, start_row, title):
        failure_imgs = imgs[label_mask][indices]
        failure_probs = probs[label_mask][indices]
        for i in range(5):
            ax = fig.add_subplot(gs[1, 0] if start_row == 1 else gs[1, 1], 
                                 fc='none', xticks=[], yticks=[])
            # Nested subplot logic for pair visualization
            sub_gs = gs[start_row, 0 if title == "Same" else 1].subgridspec(1, 5)
            sub_ax = fig.add_subplot(sub_gs[0, i])
            # Concatenate pair horizontally for display
            pair = np.hstack([failure_imgs[i, 0, 0], failure_imgs[i, 1, 0]])
            sub_ax.imshow(pair, cmap='gray')
            sub_ax.set_title(f"p={failure_probs[i]:.3f}")
            sub_ax.axis('off')
        fig.text(0.25 if title == "Same" else 0.75, 0.45, f"Top Failures: {title}", ha='center', weight='bold')

    plot_top_failures(wrong_same_idx, labels == 1, 1, "Same")
    plot_top_failures(wrong_diff_idx, labels == 0, 1, "Different")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_error_analysis()