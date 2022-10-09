from torch import nn, Tensor
import mixer

class Fredo(nn.Module):

    name = "FreDo"

    def __init__(
        self,
        d_model: int,
        num_mixers: int,
        num_input_features: int=1
        ) -> None:
        """
        Args: 

            d_model : The dimension of the lineary layers. Referred to as "O"
                      in the paper.

            num_mixers : The number of mixer layers in the model.
        """
        super().__init__()

        self.num_mixers = num_mixers

        # Create the first layer of the model
        self.linear = nn.Linear(in_features=num_input_features, out_features=d_model)

        # Create collection of Mixer modules
        self.mixers = nn.ModuleList([mixer.Mixer(d_model=d_model, num_input_features=num_input_features) for i in range(num_mixers)])

    def forward(self, x):

        x = self.linear(x)

        for mixer in self.mixers:

            x = mixer(x)

        return x


