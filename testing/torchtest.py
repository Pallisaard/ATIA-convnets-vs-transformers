import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.randn((10, 20))

L = torch.nn.Linear(20, 2)

a = a.to(device)

L = L.to(device)

b = L(a)

b = b.cpu()

print(b)
