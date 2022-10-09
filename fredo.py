from torch import nn, Tensor, fft
import mixer
import dft
import torch

class Fredo(nn.Module):

    name = "FreDo"

    def __init__(
        self,
        d_model: int,
        num_mixers: int,
        forecast_horizon: int,
        num_input_features: int=1
        ) -> None:
        """
        Args: 

            d_model : The dimension of the lineary layers. Referred to as "O"
                      in the paper.

            num_mixers : The number of mixer layers in the model.

            forecast_horizon: The desired length of the model input. Should be 48
                              if you wish to forecast 48 hours ahead for instance
        """
        super().__init__()

        self.fourier_transformer = dft.DFT()

        self.num_mixers = num_mixers

        # Create the first layer of the model
        self.linear = nn.Linear(in_features=num_input_features, out_features=d_model)

        # Create collection of Mixer modules
        self.mixers = nn.ModuleList([mixer.Mixer(d_model=d_model, num_input_features=d_model) for i in range(num_mixers)])

    def forward(self, x):

        x = self.fourier_transformer.apply_fourier_transform(x)

        # Extract real and imaginary numbers
        x_real = self.fourier_transformer.extract_real(x)

        x_imaginary = self.fourier_transformer.extract_imaginary(x)

        # Pass through linear layer
        x_real = self.linear(x_real)

        print("Shape of x_real after first linear: {}".format(x_real.shape))

        # Pass throug mixer modules
        for mixer in self.mixers:

            x_real = mixer(x_real)

        # Combine real valued model output and imaginary numbers from model input
        x = self.fourier_transformer.insert(model_output_real=x_real, model_input_imaginary=x_imaginary)

        # Inverse transform
        x = self.fourier_transformer.inverse(x)

        return x


#sequence = Tensor(24)
#sequence2 = Tensor(24)

#model_input = torch.concat((sequence, sequence2))

model_input = torch.rand(2, 24)

#model_input = torch.rand(24, 2)

#fourier = fft.fft(model_input)

#fourier.real

#fourier.imag

# TODO: Fix error
# mat1 and mat2 shapes cannot be multiplied (48x128 and 1x128) in mixer.py
model = Fredo(d_model=128, num_mixers=3, num_input_features=24)

out = model(model_input)

# it works if in_features is the same as the sequence length
linear = nn.Linear(in_features=24, out_features=128)

out = linear(model_input)

