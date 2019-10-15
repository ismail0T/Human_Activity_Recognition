import torch
import torch.nn as nn



class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        #print("init CNN_1D")

        self.drop = 0.5
        self.n_outputs = 6

        self.layer1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=9,
                    out_channels=128,  # this is the Number of filters
                    kernel_size=3,
                ),
                nn.ReLU(),
            )

        self.layer2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(self.drop)
            )
        # torch.flatten(t)

        self.fc1 = nn.Sequential(
                nn.Linear(7936, 100) # 253952 = mult of torch.Size([32, 128, 62])
                , nn.ReLU()
                ,
            )

        self.fc2 = nn.Sequential(
                nn.Linear(100, self.n_outputs)
                , nn.Softmax()
                ,
            )

        # 	model.add(Dense(100, activation='relu'))
        # 	model.add(Dense(n_outputs, activation='softmax'))

        # self.out = nn.Linear(8, 3)  #output 3 classes: Cargo, Passenger or Fishing
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = torch.flatten(out)  #  out.reshape(out.size(0), -1)  # this is equiv to flatten  1 x 253952
        # torch.Size([32, 128, 62])

        # print("before", out.shape)
        # out = torch.flatten(out)
        out = out.reshape(out.size(0), out.size(1)*out.size(2))
        # print("after", out.shape)

        out = self.fc1(out)
        out = self.fc2(out)


        # the optional argument weight (in CrossEntropyLoss) should be a 1D Tensor assigning weight to each of the classes.
        # This is particularly useful when you have an unbalanced training set.
        return out




















