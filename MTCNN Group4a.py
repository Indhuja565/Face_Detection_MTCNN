import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Load images and their ground truth boxes
images = [
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2258471435_a9c6040dcf_2061_18125152@N00.jpg",
    r"C:\\Users\\PRama\\Downloads\\Group4a\\Group4a\\952933099_d6c59f88ab_1233_20647241@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2616965735_1559d172f5_3093_10156082@N07.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2577916086_c055ef1ffe_3257_16757486@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\464972631_1c67e84f70_209_22688286@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2597934327_e1d3ac8517_3088_91636052@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2541530085_4977e8be01_3018_7238639@N04.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\463720165_af0f7a24f3_214_10145087@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2549676776_88ccd12206_3044_22422333@N07.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2340320925_e94b09ae56_3097_11099615@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2327034744_f0a79476d9_3134_27859011@N00.jpg",    
    r"C:\Users\PRama\Downloads\Group4a\Group4a\871211662_fdc7a90ea7_1317_49906406@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\824402541_76948bdfef_1276_67586024@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\360742623_173964b9aa_125_55568034@N00.jpg",
    r"C:\Users\PRama\Downloads\Group4a\Group4a\2514602521_09770e347b_2328_15799822@N02.jpg"
]

ground_truth_boxes = [
    [(698, 290, 109, 141), (482, 339, 86, 110), (223, 302, 94, 109), (602, 360, 87, 112), (385, 328, 77, 103)],
    [(270, 148, 35, 43), (136, 140, 31, 39), (206, 132, 35, 43), (331, 130, 35, 44)],
    [(70, 85, 32, 43), (281, 95, 27, 34), (399, 99, 29, 36), (195, 100, 27, 32)],
    [(314, 125, 58, 81), (647, 166, 57, 81), (560, 213, 57, 72), (419, 223, 59, 77)],
    [(643,371,118,153), (465,321,105,135), (235,227,114,155),(605,342,64,79),(778,213,142,194),(417,262,15,18),(401,264,16,22),(100,200,10,10)],
    [(234, 138, 49, 68), (378, 76, 53, 66), (705, 102, 57, 77), (536, 87, 51, 68)],
    [(194, 74, 29, 37), (375, 91, 30, 35), (100, 57, 29, 38), (284, 116, 27, 36), (315, 81, 28, 36)],
    [(481,113,53,69), (138,245,60,77), (600,159,63,74), (740,156,65,75), (100,90,70,60), (318,172,57,81)],
    [(107, 132, 37, 46), (253, 104, 35, 46), (201, 122, 39, 48), (292, 63, 32, 41), (356, 107, 36, 45)],
    [(137, 81, 16, 22), (360, 66, 17, 21), (257, 81, 18, 22), (193, 81, 18, 22), (67, 64, 19, 27), (312, 78, 19, 23)],
    [(128, 213, 25, 31), (362, 188, 29, 37), (175, 237, 22, 30), (237, 232, 24, 31), (298, 219, 23, 28)], 
    [(406,63,35,46),(272,109,31,44),(74,109,34,44),(139,76,30,35)],
    [(378,77,2,29),(120,162,18,24),(292,132,18,22),(276,204,22,28)],
    [(208,58,26,35),(348,149,25,33),(275,141,25,36),(321,72,24,31),(248,29,22,32),(371,44,20,27)],
    [(327,215,50,65),(694,39,55,76),(113,74,56,71),(514,126,45,56),(245,285,10,12)]
]

accuracies = []
recalls = []
precisions = []
execution_times = []

for img_path, gt_boxes in zip(images, ground_truth_boxes):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Start timing
    start_time = time.time()

    # Perform face detection
    faces = detector.detect_faces(image_rgb)

    # Stop timing
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)  # Store execution time

    detected_boxes = [face['box'] for face in faces]
    
    # Calculate IoU and accuracy
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gt_box in gt_boxes:
        found_match = False
        for det_box in detected_boxes:
            if calculate_iou(gt_box, det_box) > 0.1:  # Threshold for IoU
                true_positives += 1
                found_match = True
                break  # Stop after finding a match
        if not found_match:
            false_negatives += 1
    false_positives = len(detected_boxes) - true_positives

    # Calculate accuracy, recall, and precision
    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    accuracies.append(accuracy)
    recalls.append(recall)
    precisions.append(precision)

    # Draw rectangles around detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image_rgb, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Face Detection using MTCNN')
    plt.pause(2)  # Pause for 2 seconds before closing the image
    plt.close()   # Close the figure after the pause

    # Print accuracy, recall, precision, and execution time for the current image
    print(f'Accuracy for {img_path}: {accuracy:.2f}')
    print(f'Recall for {img_path}: {recall:.2f}')
    print(f'Precision for {img_path}: {precision:.2f}')
    print(f'Time taken: {execution_time:.2f} seconds')

# Calculate and print average accuracy, recall, and precision
average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
average_recall = sum(recalls) / len(recalls) if recalls else 0
average_precision = sum(precisions) / len(precisions) if precisions else 0
print(f'Average Accuracy: {average_accuracy:.2f}')
print(f'Average Recall: {average_recall:.2f}')
print(f'Average Precision: {average_precision:.2f}')

# Bar chart for speed (execution time) versus accuracy
plt.figure(figsize=(10, 5))

# Create a bar chart for execution times
bar_width = 0.4
index = np.arange(len(images))

# Plotting execution times as bars
plt.bar(index, execution_times, width=bar_width, label='Execution Time (s)', alpha=0.6, color='b')

# Overlaying accuracy using a scatter plot with dashed lines
plt.errorbar(index + bar_width, accuracies, yerr=0.01, fmt='o', label='Accuracy', alpha=0.6, color='g', linestyle='--', markersize=8)

# Display average accuracy on the graph
plt.text(len(images) / 2 - 0.5, max(max(execution_times), max(accuracies)) * 0.9,
         f'Average Accuracy: {average_accuracy:.2f}', fontsize=12, ha='center', color='black')

plt.title('MTCNN\nSpeed (Execution Time) vs Accuracy')
plt.xlabel('Image Index')
plt.ylabel('Value')
plt.xticks(index + bar_width / 2, [f'Image {i+1}' for i in range(len(images))])
plt.legend()
plt.grid()
plt.show()
