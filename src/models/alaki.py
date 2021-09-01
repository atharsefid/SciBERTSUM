import torch
# y = [1.0, 1.0 ,1.0, 1.0, 1.0]
# p = [0.0, 0.2, 0.4 , 0.8, 1.0]
# pos_weight = torch.Tensor([100])
# loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# for yy, pp in zip(y,p):
#     print(pp, yy, loss(torch.tensor([pp]), torch.tensor([yy])))


m = torch.nn.Softmax()
input = torch.randn(1, 3)
output = m(input)
print(input)
print(output, torch.sum(output))
