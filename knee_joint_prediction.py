
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'yolov5\runs\train\exp\weights\best.pt') 

# Image
im = r"knee.png"
# Load image using PIL
img = Image.open(im)

# Plot image
plt.figure(figsize=(10, 10))
plt.imshow(img)
# Inference
results = model(im)
print(results)
print(results.pandas().xyxy[0])

output_dir = r"output_images"
os.makedirs(output_dir, exist_ok=True)

# Plot bounding boxes
for box in results.xyxy[0]:
    xmin, ymin, xmax, ymax, confidence, class_id = box.tolist()
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    cropped_img = img.crop((xmin, ymin, xmax, ymax))
    
    # Save the cropped image
    cropped_img.save(os.path.join(output_dir, f"knee_{box}.png"))
    width = xmax - xmin
    height = ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(xmin, ymin, f"Class: {class_id}, Confidence: {confidence:.2f}", fontsize=12, backgroundcolor='r')

plt.axis('off')
plt.savefig(os.path.join(output_dir, f"box_{box}.png"), bbox_inches='tight')
plt.show()