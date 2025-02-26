import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# Define the file path and column names
file_path = 'train.csv'
column_names = ['text', 'target']

# Load the CSV data
dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size=32,
    select_columns=column_names,
    label_name='target',
    na_value='?',
    num_epochs=1,
    ignore_errors=True
)

# Define the TextVectorization layer
max_features = 10000  # Maximum vocabulary size
sequence_length = 100  # Maximum sequence length

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Extract the text data for adapting the vectorization layer
def extract_text(data, label):
    return data['text']

text_data = dataset.map(extract_text)
vectorize_layer.adapt(text_data)

# Apply the TextVectorization layer to the dataset
def vectorize_text(data, label):
    text = data['text']
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

vectorized_dataset = dataset.map(vectorize_text)

# Example: Print the first 5 tokenized sequences and their labels
for vectorized_text, label in vectorized_dataset.take(5):
    print(vectorized_text.numpy(), label.numpy())