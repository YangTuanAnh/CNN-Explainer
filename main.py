import cv2
import numpy as np
import os

image = cv2.imread("input_image.jpg")

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

layer_names = net.getUnconnectedOutLayersNames()

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(layer_names)

os.makedirs("MNIST", exist_ok=True)

for layer_idx, output in enumerate(layer_outputs):
    layer_dir = os.path.join("MNIST", f"{layer_names[layer_idx]}_{layer_idx+1}")
    os.makedirs(layer_dir, exist_ok=True)
    
    normalized_output = (output - output.min()) / (output.max() - output.min())

    
    for channel_idx in range(min(20, output.shape[0])):
        channel_image = output[channel_idx]
        
        channel_image_resized = cv2.resize(channel_image, (image.shape[1], image.shape[0]))
        
        normalized_channel_image = (channel_image_resized - channel_image_resized.min()) / (channel_image_resized.max() - channel_image_resized.min())
        
        heatmap = cv2.applyColorMap((normalized_channel_image * 255).astype(np.uint8), cv2.COLORMAP_JET)        
        
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

        
        channel_path = os.path.join(layer_dir, f"{channel_idx+1}.png")
        cv2.imwrite(channel_path, overlay)

print("Outputs saved.")
