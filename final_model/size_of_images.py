import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Parse the annotation line
annotation_line = "007_1_0061.png;87;85;8;12;75;77;7"
values = annotation_line.split(';')
img_path = values[0]
height = int(values[1])
width = int(values[2])
xmin = int(values[3])
ymin = int(values[4])
xmax = int(values[5])
ymax = int(values[6])

# Load the image
img = Image.open("F:\\final_model\\TRSD_dataset\\tsrd-train_img\\" + img_path)

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(img)

# Create a Rectangle patch
bbox = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')

# Add the bounding box to the plot
ax.add_patch(bbox)

# Set axis limits correctly
ax.set_xlim(0, width)
ax.set_ylim(height, 0)  # Inverted the ylim

# Show the plot
plt.show()
