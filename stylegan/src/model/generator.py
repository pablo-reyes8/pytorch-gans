from model.discriminator import * 
from model.synthesis_network import * 


class StyleGANGenerator64(nn.Module):
    """
    Generador StyleGAN v1 para 64x64 que encapsula:
    - MappingNetwork: z -> w
    - SynthesisNetwork64: w (o w por capa) -> imagen

    Soporta:
    - style mixing (probabilidad style_mixing_prob, corte aleatorio)
    - truncation trick (psi respecto a w_avg)
    """

    def __init__(self,z_dim=512,w_dim=512,
        n_mapping=8, fmap_base=512,
        ch_schedule=(512, 512, 256, 128, 64),
        lr_mul_mapping=0.01,    # mapping más lento
        use_pixelnorm=True,
        truncation_psi=1.0, # 1.0 = sin truncation
        truncation_layers=None   # si None, aplica a todas; si int, aplica a primeras L capas
    ):

        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.truncation_psi = truncation_psi
        self.truncation_layers = truncation_layers

        # Mapping z->w  con w_avg
        self.mapping = MappingNetwork(
            z_dim=z_dim, w_dim=w_dim, n_layers=n_mapping,
            use_pixelnorm=use_pixelnorm, lr_mul=lr_mul_mapping)

        # Synthesis 4→64 (10 capas "styled")
        self.synthesis = SynthesisNetwork64(
            w_dim=w_dim, fmap_base=fmap_base, ch_schedule=ch_schedule)

        self.num_layers = self.synthesis.num_layers  # 10 para 64×64

    @torch.no_grad()
    def _truncate(self, w_broadcast):
        """
        Aplica truncation trick: w' = w_avg + psi * (w - w_avg).
        - w_broadcast: [B, L, w_dim]
        - truncation_layers: si es int, solo afecta a las primeras L capas.
        """

        if self.truncation_psi >= 1.0:
            return w_broadcast
        w_avg = self.mapping.w_avg
        if self.truncation_layers is None:
            # todas las capas
            return w_avg + self.truncation_psi * (w_broadcast - w_avg)

        else:
            L = w_broadcast.size(1)
            Ltr = min(self.truncation_layers, L)
            w_out = w_broadcast.clone()
            w_out[:, :Ltr, :] = w_avg + self.truncation_psi * (w_broadcast[:, :Ltr, :] - w_avg)
            return w_out

    def forward(self,z,
        style_mixing_prob: float = 0.0,
        update_w_avg: bool = True, truncation: bool = False):

        """
        z: [B, z_dim] ~ N(0,1)
        style_mixing_prob: prob. de usar 2 estilos (z2) con corte aleatorio.
        update_w_avg: actualizar w_avg del mapping (True en train, False en eval)
        truncation: aplicar o no truncation_psi en el forward (solo inferencia)

        return: fake [B, 3, 64, 64] (logits, sin tanh)
        """
        B = z.size(0)
        device = z.device

        w1 = self.mapping(z, update_avg=update_w_avg)  # [B, w_dim]

        if style_mixing_prob > 0.0 and torch.rand(()) < style_mixing_prob:
            z2 = torch.randn(B, self.z_dim, device=device)
            w2 = self.mapping(z2, update_avg=False)
            cut = torch.randint(1, self.num_layers, (1,), device=device).item()
            w_broadcast = torch.stack([w1]*self.num_layers, dim=1)
            w_broadcast[:, cut:, :] = w2.unsqueeze(1).repeat(1, self.num_layers - cut, 1)
        else:
            # un solo estilo para todas las capas
            w_broadcast = w1.unsqueeze(1).repeat(1, self.num_layers, 1)  # [B,L,w_dim]

        if truncation:
            w_broadcast = self._truncate(w_broadcast)

        fake = self.synthesis(w_broadcast)  # [B,3,64,64]

        return fake