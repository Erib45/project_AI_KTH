import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, Dense, Embedding, Flatten, Concatenate

# Load the generated training dataset 
print("Loading data...")
data = pd.read_csv('output_data.csv')

# Encode categorical strings into integer IDs for the Embedding layers
print("Encoding data...")
link_encoder = LabelEncoder()
target_encoder = LabelEncoder()
choice_encoder = LabelEncoder()

X_links = link_encoder.fit_transform(data['links'])
X_target = target_encoder.fit_transform(data['target'])
y = choice_encoder.fit_transform(data['best_choice'])

# Persist encoders so the Flask app can use the exact same vocabularies
print("Saving encoders for future use...")
with open("link_encoder.pkl", "wb") as f: pickle.dump(link_encoder, f)
with open("target_encoder.pkl", "wb") as f: pickle.dump(target_encoder, f)
with open("choice_encoder.pkl", "wb") as f: pickle.dump(choice_encoder, f)

# 80/20 train-test split
X_train_links, X_test_links, X_train_target, X_test_target, y_train, y_test = train_test_split(
    X_links, X_target, y, test_size=0.2, random_state=42
)

# --- NEURAL NETWORK ARCHITECTURE (Functional API) ---
print("Building neural network architecture...")

# Two independent input pipelines
input_links = Input(shape=(1,), name='links_input')
input_target = Input(shape=(1,), name='target_input')

# Dynamic vocabulary sizing based on the dataset
num_unique_links = len(link_encoder.classes_)
num_unique_targets = len(target_encoder.classes_)
num_classes_output = len(choice_encoder.classes_)

# Embeddings: Compresses sparse IDs into dense 128D mathematical vectors
emb_links = Embedding(input_dim=num_unique_links, output_dim=128)(input_links)
emb_target = Embedding(input_dim=num_unique_targets, output_dim=128)(input_target)

# Flatten and merge the embeddings into a single vector
flat_links = Flatten()(emb_links)
flat_target = Flatten()(emb_target)
concat = Concatenate()([flat_links, flat_target])

# Deep dense layers for pattern recognition
dense1 = Dense(512, activation='relu')(concat)
dropout1 = Dropout(0.3)(dense1) # Prevent overfitting by randomly dropping 30% of nodes

dense2 = Dense(256, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(dense2)

# Output layer: Predicts probabilities across all possible target classes
output = Dense(num_classes_output, activation='softmax')(dropout2)

# Assemble and compile the model
model = Model(inputs=[input_links, input_target], outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement early stopping to halt training if validation loss plateaus
early_stop = EarlyStopping(
    monitor='val_loss',         
    patience=3,                 
    restore_best_weights=True   
)

# Execute training loop
print("Training the model... \n")
model.fit(
    [X_train_links, X_train_target], y_train, 
    epochs=15, 
    batch_size=256, # High batch size for fast processing on millions of rows
    validation_split=0.2
)

# Final evaluation on the holdout test set
loss, accuracy = model.evaluate([X_test_links, X_test_target], y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the weights for deployment
model.save("trained_model.keras")

# --- POST-TRAINING SANITY CHECK (CLI Testing) ---
print("Loading graph for testing... \n")
graph = nx.read_gml("wikipedia_subset_small.gml")

i = 0
while i < 10:
    start, target = random.sample(list(graph.nodes), 2)
    print(f"start: '{start}'\ntarget: '{target}'")

    try:
        path = nx.shortest_path(graph, source=start, target=target)
    except nx.NetworkXNoPath:
        continue
        
    print(f"best path:\t{path}")
    print("ai path: ")
    
    node = start
    j = 0
    AIpath = []
    
    while j < 15:
        AIpath.append(node)
        if node == target:
            break
            
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            break
            
        links_str = ",".join(neighbors)
        
        try:
            # Transform and predict using the new API structure
            links_encoded = link_encoder.transform([links_str])
            target_encoded = target_encoder.transform([target])
            
            predictions = model.predict([links_encoded, target_encoded], verbose=0)
            
            predicted_label = np.argmax(predictions, axis=1)
            node = choice_encoder.inverse_transform(predicted_label)[0]
        except Exception as e:
            print(f"Unseen node encountered: {e}")
            break
            
        j += 1
        
    if AIpath[-1] != target:
        AIpath.append("(Failed)")
        
    print(AIpath, "\n\n")
    i += 1