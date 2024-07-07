import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report


df_train = pd.read_csv("./r8/r8-train-stemmed.csv")
df_test = pd.read_csv("./r8/r8-test-stemmed.csv")

unique_values = list(set(df_train['intent'].values))
labels = {j:i for i, j in zip(range(len(unique_values)), unique_values)}

df_train['label'] = df_train['intent'].apply(lambda x: labels[x])
df_test['label'] = df_test['intent'].apply(lambda x: labels[x])

### CLASSIFICATION ###

k = 2

predicted_classes = []
accuracies = []

for idx, row_test in tqdm(enumerate(df_test.iterrows()), total=df_test.shape[0]):
    test_text = row_test[1]['text']
    test_label = row_test[1]['label']
    c_test_text = len(gzip.compress(test_text.encode()))
    distance_from_test_instance = []
    
    for row_train in df_train.iterrows():
        train_text = row_train[1]['text']
        train_label = row_train[1]['label']
        c_train_text = len(gzip.compress(train_text.encode()))
        
        train_plus_test = " ".join([test_text, train_text])
        c_train_plus_test = len(gzip.compress(train_plus_test.encode()))
        
        ncd = ((c_train_plus_test - min(c_train_text, c_test_text))
                / max(c_test_text, c_train_text))
        distance_from_test_instance.append(ncd)
        
    sorted_idx = np.argsort(np.array(distance_from_test_instance))
    top_k_class = np.array(df_train['label'])[sorted_idx[:k]]
    predicted_class = Counter(top_k_class).most_common()[0][0]
    
    predicted_classes.append(predicted_class)

    current_accuracy = np.mean(np.array(predicted_classes) == df_test['label'].values[:idx + 1])
    accuracies.append(current_accuracy)

        
print("Overall final accuracy:", np.mean(np.array(predicted_classes) == df_test['label'].values))

true_labels = df_test['label']
print("\nCLASSIFICATION REPORT:\n", classification_report(true_labels, predicted_classes))

### PLOTTING ACCURACY CURVE ###

plt.figure(figsize=(10, 6))
plt.plot(accuracies, label='Cumulative Accuracy')
plt.xlabel('Test Instance Index')
plt.ylabel('Cumulative Accuracy')
plt.title('Cumulative Accuracy Over Test Instances')
plt.legend()
plt.grid(True)
plt.show()