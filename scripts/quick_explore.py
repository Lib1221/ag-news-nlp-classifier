"""Quick script to download and explore AG News dataset."""

import logging
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download dataset
logger.info("Downloading AG News dataset...")
dataset = load_dataset('ag_news')

logger.info(f"Train samples: {len(dataset['train'])}")
logger.info(f"Test samples: {len(dataset['test'])}")

# Get labels
train_labels = dataset['train']['label']
test_labels = dataset['test']['label']

class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Train distribution
pd.Series(train_labels).value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color='steelblue'
)
axes[0].set_title('Train Set Class Distribution')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels([class_names[i] for i in range(4)], rotation=45)

# Test distribution
pd.Series(test_labels).value_counts().sort_index().plot(
    kind='bar', ax=axes[1], color='lightgreen'
)
axes[1].set_title('Test Set Class Distribution')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels([class_names[i] for i in range(4)], rotation=45)

plt.tight_layout()
plt.savefig('logs/class_distribution.png', dpi=150, bbox_inches='tight')
logger.info("Class distribution plot saved to logs/class_distribution.png")

# Print statistics
logger.info("\n=== Dataset Statistics ===")
total = len(train_labels) + len(test_labels)
logger.info(f"Total samples: {total}")
logger.info(f"Train: {len(train_labels)} ({len(train_labels)/total*100:.1f}%)")
logger.info(f"Test: {len(test_labels)} ({len(test_labels)/total*100:.1f}%)")

logger.info("\nClass Distribution (Train):")
for class_id, class_name in class_names.items():
    count = (pd.Series(train_labels) == class_id).sum()
    logger.info(f"  {class_name}: {count} samples ({count/len(train_labels)*100:.1f}%)")

# Show sample texts
logger.info("\n=== Sample Texts ===")
for i in range(3):
    text = dataset['train'][i]['text']
    label = class_names[dataset['train'][i]['label']]
    logger.info(f"\nSample {i+1} ({label}):")
    logger.info(f"  {text[:100]}...")
