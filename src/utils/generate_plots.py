import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob
from PIL import Image

# Get the latest metrics file from logs/train/runs/*/csv_logs/version_*/metrics.csv
metrics_files = glob.glob('logs/train/runs/*/logs/csv_logs/version_*/metrics.csv')
if not metrics_files:
    print("Error: No metrics files found.")
    sys.exit(1)
latest_metrics_file = max(metrics_files, key=os.path.getctime)  # Get the latest file

# Load the metrics data
try:
    data = pd.read_csv(latest_metrics_file)
except FileNotFoundError:
    print(f"Error: The file {latest_metrics_file} was not found.")
    sys.exit(1)

# Check if the necessary columns exist
if not all(col in data.columns for col in ['epoch', 'train/acc', 'train/loss']):
    print("Error: Required columns are missing in the metrics file.")
    sys.exit(1)

# Generate test metrics table report in test_metrics.md
with open('test_metrics.md', 'w') as f:
    f.write("## Test Metrics\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Test Accuracy | {data['test/acc'].iloc[-1]} |\n")  # Assuming last entry is the latest
    f.write(f"| Test Loss | {data['test/loss'].iloc[-1]} |\n")  # Assuming last entry is the latest

# Plot Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['train/acc'], marker='o', color='b', label='Train Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(data['epoch'])
plt.legend()
plt.savefig('train_acc_plot.png')
plt.close()

# plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['val/acc'], marker='o', color='b', label='Validation Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(data['epoch'])
plt.legend()
plt.savefig('val_acc_plot.png')
plt.close()

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['train/loss'], marker='o', color='b', label='Train Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(data['epoch'])
plt.legend()
plt.savefig('train_loss_plot.png')
plt.close()

# val loss
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['val/loss'], marker='o', color='b', label='Validation Loss')
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(data['epoch'])
plt.legend()
plt.savefig('val_loss_plot.png')
plt.close()

# Create Collage of Predictions
predictions_dir = './predictions'
images = []
for img_path in glob.glob(f'{predictions_dir}/*/*.png'):
    img = Image.open(img_path)
    images.append(img)

# Create a collage
if images:
    # Define collage size
    collage_width = 800  # Adjust as needed
    collage_height = 800  # Adjust as needed
    collage_image = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

    # Calculate dimensions for each image in the collage
    num_images = len(images)
    grid_size = int(num_images**0.5) + 1  # Determine grid size
    img_width = collage_width // grid_size
    img_height = collage_height // grid_size

    for index, img in enumerate(images):
        img = img.resize((img_width, img_height), Image.LANCZOS)
        x = (index % grid_size) * img_width
        y = (index // grid_size) * img_height
        collage_image.paste(img, (x, y))

    collage_image.save('predictions_collage.png')

# Create Predictions Markdown
with open('predictions.md', 'w') as f:
    f.write("### Predictions Collage\n")
    f.write(f"<img src='predictions_collage.png' width='800' />\n")
