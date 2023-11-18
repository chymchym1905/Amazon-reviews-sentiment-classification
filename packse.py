import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

torch.manual_seed(12)
data = [torch.tensor([16]),
        torch.tensor([204, 185, 65, 345]), 
        torch.tensor([204, 987]),
        torch.tensor([85, 4, 23])]

lengths = [d.size(0) for d in data]
# print(lengths)

padded_data = pad_sequence(data, batch_first=True, padding_value=0)
print(padded_data)
mask = (padded_data != 0).float()
print(mask.sum(1))
embedding = nn.Embedding(1000, 300, padding_idx=0)
embeded_data = embedding(padded_data)
print(embeded_data.shape)
packed_data = pack_padded_sequence(embeded_data, lengths, batch_first=True, enforce_sorted=False)
print(packed_data)
lstm = nn.LSTM(300, 512, batch_first=True)
o, (h, c) = lstm(packed_data)

# (h, c) is the needed final hidden and cell state, with index already restored correctly by LSTM.
# but o is a PackedSequence object, to restore to the original index:

unpacked_o, unpacked_lengths = pad_packed_sequence(o, batch_first=True)
# now unpacked_o, (h, c) is just like the normal output you expected from a lstm layer.

# print(f'{unpacked_o}\nBBBBB\n {unpacked_lengths}')
print(f'{h}\nAAAAA\n {c}')
fc = nn.Linear(512,2)
softmax = nn.LogSigmoid()
criterion = nn.BCEWithLogitsLoss()
print(h.shape)
# out = fc(h)
# print(out, len(out))
# o = fc(unpacked_o)
# print(o, len(o))
# a = o.argmax(1)
# print(a)
o = fc(h.squeeze_(0))
o = softmax(o)
print(o, len(o))
i, res = torch.max(o, 1)
print(res)
print([res == torch.tensor([[1],[0],[1],[0]], dtype=torch.float).squeeze(1)])
# loss = criterion(o, torch.tensor([[1],[0],[1],[0]], dtype=torch.float))
# print(loss)