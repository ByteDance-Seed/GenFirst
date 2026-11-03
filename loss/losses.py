"""This files contains training loss implementation.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast
from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator, weights_init


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class ReconstructionLoss_Stage1(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        loss_config = config.losses
        self.quantizer_weight = loss_config.quantizer_weight
        self.target_codebook_size = 1024

    def forward(self,
                target_codes: torch.Tensor,
                reconstructions: torch.Tensor,
                quantizer_loss: torch.Tensor,
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        return self._forward_generator(target_codes, reconstructions, quantizer_loss)

    def _forward_generator(self,
                           target_codes: torch.Tensor,
                           reconstructions: torch.Tensor,
                           quantizer_loss: Mapping[Text, torch.Tensor],
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        reconstructions = reconstructions.contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        batch_size = reconstructions.shape[0]
        reconstruction_loss = loss_fct(reconstructions.view(batch_size, self.target_codebook_size, -1),
                                        target_codes.view(batch_size, -1))
        total_loss = reconstruction_loss + \
            self.quantizer_weight * quantizer_loss["quantizer_loss"]

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss["quantizer_loss"]).detach(),
            commitment_loss=quantizer_loss["commitment_loss"].detach(),
            codebook_loss=quantizer_loss["codebook_loss"].detach(),
        )

        return total_loss, loss_dict


class ReconstructionLoss_Stage2(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator(
            input_nc=3,
            n_layers=3,
            use_actnorm=False
        ).apply(weights_init)

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                latents_scale=None,
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step, latents_scale=latents_scale)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        elif mode == "generator_alignment":
            return self._forward_generator_alignment(inputs, reconstructions, extra_result_dict, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           latents_scale=None,
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           latents_scale=None,
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step with REPA alignment."""
        return NotImplementedError("This function should be implemented in the subclass.")

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict


class ReconstructionLoss_Single_Stage(ReconstructionLoss_Stage2):
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        loss_config = config.losses
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        
        if self.quantize_mode == "vae":
            self.kl_weight = loss_config.get("kl_weight", 1e-6)
            logvar_init = loss_config.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)
            # Latent regularizer applied to the Gaussian posterior. "kl" is the
            # standard VAE KL-to-N(0, I); "entropy" instead *maximizes* the
            # posterior entropy (total_loss += -kl_weight * entropy), which is
            # used when the diffusion gradient flows into the VAE to prevent
            # latent collapse. May be overridden by the training script.
            self.latent_reg = loss_config.get("latent_reg", "kl")
            # Auxiliary (small) KL added *on top of* the entropy regularizer.
            # Entropy only maximizes posterior variance (logvar) and does NOT
            # constrain the posterior `mean`; under the end-to-end diffusion
            # gradient the encoder mean can drift unbounded -> NaN. The KL's
            # `mean^2` term pulls the mean back toward 0. Kept tiny so it does
            # not fight the entropy term on the variance. 0 disables it.
            self.aux_kl_weight = loss_config.get("entropy_aux_kl_weight", 1e-6)

        # coefficient for the projection alignment loss
        self.proj_coef = loss_config.get("proj_coef", 0.0)

    def _latent_reg_term(self, posteriors, latents_scale=None):
        """Return (weighted_reg_term, raw_reg_value, raw_aux_kl) for the regularizer.

        ``weighted_reg_term`` is what gets added to ``total_loss``;
        ``raw_reg_value`` is the unweighted scalar (KL or entropy);
        ``raw_aux_kl`` is the unweighted auxiliary KL (entropy mode only, else 0).

        ``latents_scale`` (per-channel ``1/sqrt(var_c)`` from the SiT BN) is only
        used by the entropy regularizer: it reweights the per-channel entropy so
        its gradient w.r.t. ``logvar`` matches the per-channel BN factor that the
        diffusion gradient already carries. The global strength is still set by
        ``kl_weight`` (``--latent-reg-weight``). KL ignores this argument.
        """
        if getattr(self, "latent_reg", "kl") == "entropy":
            ent = posteriors.entropy(weight=latents_scale)
            ent = torch.sum(ent) / ent.shape[0]
            # Auxiliary KL (mainly its mean^2 term) constrains the posterior
            # mean so it cannot drift unbounded under the diffusion gradient.
            kl = posteriors.kl()
            kl = torch.sum(kl) / kl.shape[0]
            # Maximize entropy -> subtract; add the small KL on top.
            reg_term = -self.kl_weight * ent + self.aux_kl_weight * kl
            return reg_term, ent, kl
        else:
            kl = posteriors.kl()
            kl = torch.sum(kl) / kl.shape[0]
            zero = torch.zeros((), device=kl.device, dtype=kl.dtype)
            return self.kl_weight * kl, kl, zero

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           latents_scale=None,
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0.
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        if self.quantize_mode == "vq":
            # Compute quantizer loss.
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
        elif self.quantize_mode == "vae":
            # Compute the latent regularizer (KL or negative-entropy).
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            posteriors = extra_result_dict
            reg_term, reg_value, aux_kl_value = self._latent_reg_term(posteriors, latents_scale)
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + reg_term
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                # `kl_loss` key holds the (signed) weighted latent-reg term so
                # downstream logging keeps working for both kl/entropy modes.
                kl_loss=reg_term.detach(),
                latent_reg_value=reg_value.detach(),
                # Raw (unweighted) auxiliary KL added in entropy mode (0 in kl mode).
                aux_kl_value=aux_kl_value.detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor).to(generator_loss.device),
                d_weight=torch.tensor(d_weight).to(generator_loss.device),
                gan_loss=generator_loss.detach(),
            )
        else:
            raise NotImplementedError

        return total_loss, loss_dict

    def _forward_generator_alignment(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step with REPA alignment."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0.
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Projection alignment loss, we suppose all inputs are provided in `extra_result_dict`
        zs_tilde = extra_result_dict["zs_tilde"]  # list of [B, N, C]
        zs = extra_result_dict["zs"]              # list of [B, N, C]

        proj_loss = torch.tensor(0., device=inputs.device)
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        if self.quantize_mode == "vq":
            # Compute quantizer loss.
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
                + self.proj_coef * proj_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
                proj_loss=proj_loss.detach(),
            )
        elif self.quantize_mode == "vae":
            # Compute kl loss.
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            # NOTE: because we need other inputs, so we put posteriors as one element in extra_result_dict
            posteriors = extra_result_dict["posteriors"]
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
                + self.proj_coef * proj_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                kl_loss=(self.kl_weight * kl_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor).to(generator_loss.device),
                d_weight=torch.tensor(d_weight).to(generator_loss.device),
                gan_loss=generator_loss.detach(),
                proj_loss=proj_loss.detach(),
            )
        else:
            raise NotImplementedError

        return total_loss, loss_dict


class MLMLoss(torch.nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.label_smoothing = config.losses.label_smoothing
        self.loss_weight_unmasked_token = config.losses.loss_weight_unmasked_token
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                   reduction="none")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(inputs, "b n c -> b c n")
        loss = self.criterion(inputs, targets)
        weights = weights.to(loss)
        loss_weights = (1.0 - weights) * self.loss_weight_unmasked_token + weights # set 0 to self.loss_weight_unasked_token
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
        # we only compute correct tokens on masked tokens
        correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(dim=1) / (weights.sum(1) + 1e-8)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
    

class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous() # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
