import numpy as np
from torch import fft, Tensor

class DFT:

    def __init__(self) -> None:
        pass

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