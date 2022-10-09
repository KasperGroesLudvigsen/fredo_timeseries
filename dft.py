import numpy as np
from torch import fft, Tensor

class DFT:

    def __init__(self) -> None:
        pass

    def discard_zeros(x: Tensor, input_length: int):
        """
        According to the paper, they "only use positive frequency components" 
        (p.5)
        """

        # As per the paper: If the input length is even
        if input_length % 2 == 0:
            
            last_real_idx = input_length/2

            last_imaginary_idx = (input_length/2)-1

        # If input length is odd
        else:

            last_real_idx = (input_length-1)/2

            last_imaginary_idx = (input_length-1)/2

        # Assuming batch first
        real = x[:, 0:last_real_idx].real

        imaginary = x[:, 1:last_imaginary_idx].imag

        x_cat = torch.concat((real, imaginary))

        # To do: Figure out how to "concatenate them into a real-valued vector"




    def apply_fourier_transform(self, x):
        """
        Apply fast fourier transform on x and return real values of the transformed
        x.
        """
        
        # Perform Fourier Transform - returns array of complex numbers
        x = fft.fft(x)

        return x

        #return x.real

    def extract_real(self, x: Tensor):
        """
        Extract real values from fourier transformed x

        Args:

            x: A fourier transformed sequence
        """
        return x.real 

    def extract_imaginary(self, x: Tensor):
        """
        Extract imaginary values from fourier transformed x

        Args: 

            x: A fourier transformed sequence
        """
        return x.imag

    def insert(self, model_output_real: Tensor, model_input_imaginary: Tensor):
        """
        Combine the real valued model output with the imginary part extracted from 
        the model input

        Args: 

            model_output_real: The output of the FreDo model while still in the
                               frequency domain.

            model_input_imaginary: The imaginary part of the model input AFTER
                                   fourier transform and before being given as
                                   input to the model.

        """

        # multiply by 1.0j to convert model_input to complex number again
        # Not sure I need to multiply by 1.0j because model_input_imaginary
        # will already be in its imaginary form
        #x = model_output_real + model_input_imaginary * 1.0j 

        x = model_output_real + model_input_imaginary

        return x

    def inverse(self, x: Tensor):

        """
        Apply inverse fourier transform to bring x back from frequency domain to 
        time domain.
        """
        
        return fft.ifft(x)