import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_with_box(ax, image_path, xmin, ymin, xmax, ymax):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    ax.imshow(image)

    # Create a rectangle patch
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')

    # Add the rectangle to the Axes
    ax.add_patch(rect)

# Example usage
ann_path = r"F:\final_model\TRSD_dataset\TSRD-Train Annotation\TsignRecgTrain4170Annotation.txt"
with open(ann_path, "r") as f:
    lines = f.readlines()

    # Display images in batches of 50
    batch_size = 50
    total_images = len(lines)

    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        
        # Create a grid of subplots for the current batch
        rows = (batch_end - batch_start) // 10 + 1
        cols = min(batch_end - batch_start, 10)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

        for i, line in enumerate(lines[batch_start:batch_end]):
            line_comp = list(map(str, line.strip().split(';')))
            img = line_comp[0]
            xmin, ymin, xmax, ymax = map(int, (line_comp[3], line_comp[4], line_comp[5], line_comp[6]))
            image_path = r"F:\\final_model\\TRSD_dataset\\tsrd-train_img\\" + str(img)

            # Calculate the subplot index
            row_idx = i // cols
            col_idx = i % cols

            # Display the image with bounding box on the corresponding subplot
            display_image_with_box(axs[row_idx, col_idx], image_path, xmin, ymin, xmax, ymax)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    
