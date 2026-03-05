import random
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Load the data file 

data = pd.read_csv('output_data.csv')

#Create structure for data and correct answer
X = data[['links', 'target']]
y = data['best_choice']

#Encode the data to numbers so that the neural network can understand it
link_encoder = LabelEncoder()
target_encoder = LabelEncoder()
choice_encoder = LabelEncoder()

X['links'] = X['links'].apply(lambda x: x.split(','))  #Split the links
X['links'] = X['links'].apply(lambda x: ' '.join(x))  #Join as a single string

X['links'] = link_encoder.fit_transform(X['links'])
X['target'] = target_encoder.fit_transform(X['target'])
y = choice_encoder.fit_transform(y)

#One hot encode the data (turn the numbers into binary)
one_hot_encoder = OneHotEncoder()
X_encoded = one_hot_encoder.fit_transform(X).toarray()

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#Create the neural network model
model = Sequential()
#Add the first layer with 64 neurons and relu activation function
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#Add the second layer (hidden layer) with 32 neurons and relu activation function
model.add(Dense(32, activation='relu'))
#Add the output layer, len(np.unique(y)) represents all the unique target classes i.e all the possible outputs since we made sure all nodes are the target atleast once in training
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer

#Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
print("Training the model... \n")
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

#Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

#The model can be saved and used in another program, but for now we will just use it in this program
model.save("trained_model.keras")

#Load the graph
print("Loading graph... \n")
graph = nx.read_gml("wikipedia_subset_small.gml")

def find_shortest_path(graph, start, target):
    #Find the shortest path between two nodes in the graph using networkx built in function
    try:
        return nx.shortest_path(graph, source=start, target=target)
    except nx.NetworkXNoPath:
        return []

i = 0
#Test the model on 10 random paths and compare it to the shortest path
while i < 10:
    start, target = random.sample(list(graph.nodes), 2)
    print(f"start: '{start}'\ntarget: '{target}'")

    path = find_shortest_path(graph, start, target)
    if(path == []):
        continue
    print(f"best path:\t{path}")

    print("ai path: ")
    node = start
    j = 0
    AIpath = []
    while j < 10:
        AIpath.append(node)
        neighbors = list(graph.neighbors(node))
        new_data = pd.DataFrame({
            'links': [",".join(neighbors)],
            'target': [target]
        })

        # Encode the new data
        new_data['links'] = new_data['links'].apply(lambda x: x.split(','))
        new_data['links'] = new_data['links'].apply(lambda x: ' '.join(x))
        new_data['links'] = link_encoder.transform(new_data['links'])
        new_data['target'] = target_encoder.transform(new_data['target'])

        # One hot encode the new data
        new_data_encoded = one_hot_encoder.transform(new_data).toarray()

        predictions = model.predict(new_data_encoded, verbose=0)
        predicted_label = np.argmax(predictions, axis=1)
        predicted_best_choice = choice_encoder.inverse_transform(predicted_label)

        node = predicted_best_choice[0]
        if node == target:
            break
        j += 1
    AIpath.append(node)
    print(AIpath)
    print("\n\n")
    i += 1
