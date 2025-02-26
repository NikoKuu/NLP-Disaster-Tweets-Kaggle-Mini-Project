import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd


# Load the text data
# file_path = r'C:\Users\kuusnin\tempwork\temp\test_text.txt'
# dataset = tf.data.TextLineDataset(file_path)

train_df = pd.read_csv('train.csv')
dataset = tf.data.Dataset.from_tensor_slices(train_df.text.values)

# Define the TextVectorization layer
max_features = 10000  # Maximum vocabulary size
sequence_length = 100  # Maximum sequence length

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Adapt the layer to the dataset
vectorize_layer.adapt(dataset)

# Apply the TextVectorization layer to the dataset
def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)

vectorized_dataset = dataset.map(vectorize_text)

# Example: Print the first 5 tokenized sequences
for vectorized_text in vectorized_dataset.take(5):
    print(vectorized_text.numpy())


target_ds = tf.data.Dataset.from_tensor_slices(train_df.target.values)

train_ds = tf.data.Dataset.zip((vectorized_dataset, target_ds))

print('done')
