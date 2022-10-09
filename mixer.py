from torch import nn, Tensor

class Mixer(nn.Module):

    name = "Mixer"

    def __init__(
        self, 
        d_model: int,
        num_input_features: int=1
        ) -> None:

        """
        Args: 

            d_model : The dimension of the linear layers. Referred to as O in the
                      paper. 

            num_input_features : The number of variables in the model input. 
                                 Should be 1 if univariate.

        """
        super().__init__()

        self.linear1 = nn.Linear(in_features=num_input_features, out_features=d_model)
        self.linear2 = nn.Linear(in_features=num_input_features, out_features=d_model)
        self.activation = nn.ReLU()

    def forward(self, model_input):
        """
        Apply linear 1 and linear 2 with ReLU in between
        """
        return self.linear2(self.activation(self.linear1(model_input)))