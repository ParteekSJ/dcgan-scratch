import torch
import ipdb
from constants import *
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import ipdb

from utils.utils import (
    init_setting,
    show_tensor_images,
    plot_loss_curves,
    weights_init,
)
from model.generator import Generator, create_noise_vector
from model.discriminator import Discriminator
from logger.logger import get_logger, setup_logging


if __name__ == "__main__":
    experiment_dir, checkpoint_dir, image_dir = init_setting()
    dataloader = DataLoader(
        dataset=MNIST(root="data", download=False, transform=TRANSFORMS),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    setup_logging(save_dir=experiment_dir)

    logger = get_logger(name="train")  # log message printing

    generator = Generator(im_chan=1, z_dim=Z_DIM, num_gen_filter=HIDDEN_DIM).to(DEVICE)
    discriminator = Discriminator(im_chan=1, num_disc_filter=HIDDEN_DIM).to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizerD = torch.optim.Adam(params=discriminator.parameters(), lr=D_LR)
    optimizerG = torch.optim.Adam(params=generator.parameters(), lr=G_LR)

    G_LOSSES, D_LOSSES = [], []
    for epoch in range(NUM_EPOCHS):
        generator_losses, discriminator_losses = [], []
        for idx, (images, labels) in enumerate(dataloader):
            generator.train()
            discriminator.train()

            # batch_size_real_images = len(images)
            batch_size_real_images = images.shape[0]

            ## TRAIN DISCRIMINATOR
            optimizerD.zero_grad()  # Zero out the gradients.
            images = images.to(DEVICE)  # Real images

            noise = create_noise_vector(
                n_samples=BATCH_SIZE,
                input_dim=Z_DIM,
            ).to(DEVICE)
            fake = generator(noise)  # Synthesizing Samples.

            # Discriminator's Predictions on Real and Fake Images.
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(images)

            disc_fake_loss = CRITERION(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = CRITERION(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            # Backpropagate & Update Weights
            disc_loss.backward(retain_graph=True)
            discriminator_losses.append(disc_loss.item())
            optimizerD.step()

            ## TRAIN GENERATOR
            optimizerG.zero_grad()
            disc_real_pred_with_gradients = discriminator(fake)
            gen_loss = CRITERION(
                disc_real_pred_with_gradients,
                torch.ones_like(disc_real_pred_with_gradients),
            )
            gen_loss.backward()
            generator_losses.append(gen_loss.item())
            optimizerG.step()

            if idx % DISPLAY_STEP == 0 and idx > 0:
                # Calculate Generator & Discriminator Mean Loss for the latest display steps (i.e., last 50 steps)
                gen_mean = sum(generator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                disc_mean = sum(discriminator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                logger.info(f"Epoch {epoch}: | Step: {idx} | Gen Loss: {gen_mean} | Disc Loss: {disc_mean}")

        checkpoint = {
            "epoch": epoch,
            "gen_state_dict": generator.state_dict(),
            "disc_state_dict": discriminator.state_dict(),
            "gen_optimizer": optimizerG.state_dict(),
            "disc_optimizer": optimizerD.state_dict(),
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
        }  # save state dictionary
        torch.save(checkpoint, f"{checkpoint_dir}/model.pth")

        G_LOSSES.append(sum(generator_losses[:]) / len(dataloader))
        D_LOSSES.append(sum(discriminator_losses[:]) / len(dataloader))

        show_tensor_images(fake, show=False, plot_name=f"{image_dir}/epoch-{epoch}-fake.png")
        show_tensor_images(images, show=False, plot_name=f"{image_dir}/epoch-{epoch}-real.png")

    # Plotting the GAN loss curves
    plot_loss_curves(G_LOSSES, D_LOSSES, checkpoint_dir)
