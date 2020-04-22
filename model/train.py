# Script to train Bidirectional LSTM model.
# Note: I ran this on Google Colab to leverage GPU power.

import utils.data_process as dp
import torch
import torch.nn as nn
import torch.optim as optim
from seqeval.metrics import f1_score, accuracy_score, classification_report

from model.model import BidirectionalLSTM, predict
from utils.data_process import generate_batches

###########################################
#       Data processing                   #
###########################################

# Read in and process data
path = "../data/CONLL2003/"

train_tokens, train_tags = dp.parse_file(path + "train.txt")
valid_tokens, valid_tags = dp.parse_file(path +"valid.txt")
test_tokens, test_tags = dp.parse_file(path + "test.txt")

# Sort the tokens for easier batching later
train_tokens_sorted = sorted(train_tokens, key=len)
train_tags_sorted = sorted(train_tags, key=len)

# Create index lookup. Include UNK for unknown words, PAD for padding
word2idx_train = dp.index_dicts(train_tokens, special_tokens=["<UNK>", "<PAD>"])
tag2idx_train = dp.index_dicts(train_tags, special_tokens=["<PAD>"])

# Convert tokens to index number and store in a tensor
train_tokens_indexed = dp.convert_to_index(train_tokens_sorted, word2idx_train)
train_tags_indexed = dp.convert_to_index(train_tags_sorted, tag2idx_train)

# Convert labels to tensor
labels = torch.tensor([i for i in tag2idx_train.values()], dtype=torch.long)

# Label to tag lookup
idx2tag_train = {}
for k, v in tag2idx_train.items():
    idx2tag_train[v] = k

###########################################
#       Train the model                   #
###########################################

# Model parameters
batch_size = 32
vocab_size = len(word2idx_train)
tag_size = len(labels)
embedding_dim = 128
n_hidden = 256
dropout = 0.5
learning_rate = 0.001
n_epochs = 50

# Model time!
bilstm = BidirectionalLSTM(vocab_size, tag_size, embedding_dim, n_hidden, batch_size, dropout)

# Calculate loss with cross entropy. Ignore index for <PAD> token.
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Use Adam optimizer
optimizer = optim.Adam(bilstm.parameters(), lr=learning_rate)

# Set model to "train" mode
bilstm.train()

for epoch in range(n_epochs):
    print("Epoch", epoch + 1)
    running_loss = 0.0

    for batch_x, batch_y in generate_batches(batch_size, train_tokens_indexed, train_tags_indexed):

        # collect outputs for loss calc
        out_tensor = torch.zeros([batch_x.shape[0], batch_x.shape[-1], tag_size])

        optimizer.zero_grad()
        for i in range(len(batch_x)):
            output = bilstm(batch_x[i])

            out_tensor[i] = output
            output_arg_max = [torch.argmax(i).item() for i in output]

        loss = criterion(out_tensor.permute(0, 2, 1), batch_y)

        # Keep track of loss
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss)

        loss.backward()

        optimizer.step()

    # Evaluate at end of each epoch
    y_pred = predict(train_tokens_indexed, bilstm, idx2tag_train)

    print(f"Epoch {epoch + 1} Loss:", running_loss)
    print(f"Epoch {epoch + 1} F1:", f1_score(train_tags, y_pred))
    print(f"Epoch {epoch + 1} Acc:", accuracy_score(train_tags, y_pred))

###########################################
#       Evaluate on validation set        #
###########################################

# Create tensors of indexed tokens for validation data
valid_tokens_indexed = dp.convert_to_index(valid_tokens, word2idx_train)
valid_tags_indexed = dp.convert_to_index(valid_tags, tag2idx_train)

# Set model to evaluation mode - we won't update it during this phase.
bilstm.eval()

# Since we're evaluating, we shouldn't keep track of the gradients.
with torch.no_grad():
    y_pred = predict(valid_tokens_indexed, bilstm, idx2tag_train)

print("F1 score:", f1_score(valid_tags, y_pred))
print("Accuracy:", accuracy_score(valid_tags, y_pred))
print(classification_report(valid_tags, y_pred))

# Assuming that looks decent, save the model for app deployment
torch.save(bilstm.state_dict(), "model/bilstm.pt")

#######################################################
# TODO: incorporate vocab from validation into training

