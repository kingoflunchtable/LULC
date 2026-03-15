import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Your real results from the terminal
cm = np.array([
    [746, 0, 14, 17, 0, 19, 22, 0, 20, 20],
    [0, 852, 18, 0, 0, 16, 0, 0, 5, 11],
    [5, 5, 791, 13, 15, 28, 26, 16, 9, 3],
    [25, 2, 54, 488, 48, 15, 43, 11, 54, 0],
    [0, 0, 28, 24, 635, 0, 11, 29, 1, 0],
    [5, 13, 61, 13, 0, 472, 20, 0, 29, 10],
    [30, 0, 149, 31, 23, 9, 508, 5, 16, 0],
    [0, 0, 33, 7, 26, 0, 0, 851, 0, 0],
    [23, 8, 43, 104, 5, 32, 14, 4, 492, 15],
    [0, 4, 0, 0, 0, 7, 0, 0, 2, 897]
])

classes = ['AnnualCrop', 'Forest', 'Herbaceous', 'Highway', 'Industrial', 
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Final LULC Classification Report (Confusion Matrix)')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.savefig('models/confusion_matrix_visual.png')
print("✅ Visual Report saved to models/confusion_matrix_visual.png")