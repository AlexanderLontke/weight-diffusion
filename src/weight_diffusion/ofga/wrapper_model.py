from torch import nn


class WrapperModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            diffusion_model):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.diffusion_model = diffusion_model

    def forward(self, models):
        # TODO revise
        latent_representation = self.encoder(models)
        x = self.diffusion_model(latent_representation)
        x = self.decoder(x)
        return x
