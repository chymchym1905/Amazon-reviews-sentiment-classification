import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

torch.manual_seed(12)
data = [torch.tensor([1]),
        torch.tensor([2, 3, 4, 5]), 
        torch.tensor([6, 7]),
        torch.tensor([8, 9, 10])]

lengths = [d.size(0) for d in data]
# print(lengths)

padded_data = pad_sequence(data, batch_first=True, padding_value=0)
print(padded_data)
mask = (padded_data != 0).float()
print(mask.sum(1))
embedding = nn.Embedding(20, 5, padding_idx=0)
embeded_data = embedding(padded_data)

packed_data = pack_padded_sequence(embeded_data, lengths, batch_first=True, enforce_sorted=False)
# print(packed_data)
lstm = nn.LSTM(5, 5, batch_first=True)
o, (h, c) = lstm(packed_data)

# (h, c) is the needed final hidden and cell state, with index already restored correctly by LSTM.
# but o is a PackedSequence object, to restore to the original index:

unpacked_o, unpacked_lengths = pad_packed_sequence(o, batch_first=True)
# now unpacked_o, (h, c) is just like the normal output you expected from a lstm layer.

print(f'{unpacked_o}\nBBBBB\n {unpacked_lengths}')
print(f'{h}\nAAAAA\n {c}')
fc = nn.Linear(5,1)
criterion = nn.BCEWithLogitsLoss()
# out = fc(h)
# print(out, len(out))
# o = fc(unpacked_o)
# print(o, len(o))
# a = o.argmax(1)
# print(a)
o = fc(h.squeeze(0))
print(o, len(o))
loss = criterion(o, torch.tensor([[1],[0],[1],[0]], dtype=torch.float))
print(loss)