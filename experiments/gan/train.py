import argparse
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from models import gan
from mlx.utils import tree_flatten
from PIL import Image
import os

def load_anime_faces(directory, img_size=(64, 64)):
    """
    Load anime faces from a directory, resize to img_size, and normalize pixel values.
    """
    image_files = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    images = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        img = np.array(img) / 127.0 - 1  # Normalize pixel values to [0, 1]
        images.append(img)
    images = np.stack(images)
    return images

def anime_faces_batch_iterator(images, batch_size):
    """
    Yield batches of anime faces.
    """
    num_samples = len(images)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield images[batch_indices]

def grid_image_from_batch(image_batch, num_rows):
    """
    Generate a grid image from a batch of images.
    Assumes input shape is (B, H, W, C).
    """
    B, H, W, C = image_batch.shape
    num_cols = (B + num_rows - 1) // num_rows  # Ensure enough columns for all images

    # Calculate grid image size
    grid_height = num_rows * H
    grid_width = num_cols * W

    # Normalize and convert to uint8
    image_batch = ((image_batch+1)*127).astype(np.uint8)

    # Create an empty array for the grid image
    grid_image = np.zeros((grid_height, grid_width, C), dtype=np.uint8)

    # Fill the grid with images
    for idx, img in enumerate(image_batch):
        row = idx // num_cols
        col = idx % num_cols
        grid_image[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = img

    return Image.fromarray(grid_image)

def generate_and_save_images(generator, noise, epoch, save_dir, num_samples=64):
    """
    Generate images using the generator and save as a grid image.
    """
    fake_images = generator(noise)
    fake_images = np.array(fake_images)
    grid_image = grid_image_from_batch(fake_images, num_rows=8)
    grid_image.save(save_dir / f'generated_{epoch:03d}.png')

def loss_fn_dis(model_dis, model_gen, real_data, noise):
    real_output = model_dis(real_data)
    real_labels = mx.full([real_output.shape[0]], 1.0)
    loss_real = nn.losses.binary_cross_entropy(real_output, real_labels)

    fake_data = model_gen(noise)
    fake_output = model_dis(fake_data)
    fake_labels = mx.full([real_output.shape[0]], 0.0)
    loss_fake = nn.losses.binary_cross_entropy(fake_output, fake_labels)

    return loss_real + loss_fake

def loss_fn_gen(model_dis, model_gen, noise):
    fake_data = model_gen(noise)
    fake_output= model_dis(fake_data)
    real_labels = mx.full([fake_output.shape[0]], 1.0)
    loss = nn.losses.binary_cross_entropy(fake_output, real_labels)

    return loss

def main(args):
    nf = 64
    nc = 3
    nz = 128

    train_images = load_anime_faces("datasets/anime_faces_cleaned", nc)
    train_iter = anime_faces_batch_iterator(train_images, args.batch_size)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_dis = gan.Discriminator(nf, nc)
    # model_dis.load_weights("experiments/gan/take_0/weights_dis.npz")
    mx.eval(model_dis)

    model_gen = gan.Generator(nz, nf, nc)
    # model_gen.load_weights("experiments/gan/take_0/weights_gen.npz")
    mx.eval(model_gen)

    # def init_weights(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #         nn.init.glorot_normal(m.weight)  # Initialize weights for Conv layers
    #     elif isinstance(m, nn.BatchNorm):
    #         m.weight.data.fill_(1.0)  # Batch norm layers usually initialize with 1.0 for scale
    #         m.bias.data.zero_()  # and 0 for biases
    #
    # # Apply the initialization only to the appropriate layers
    # model_dis.apply(init_weights)
    # model_gen.apply(init_weights)


    n_params_d = sum(x.size for _, x in tree_flatten(model_gen.trainable_parameters()))
    n_params_g = sum(x.size for _, x in tree_flatten(model_dis.trainable_parameters()))
    print(f"Number of trainable discriminator params: {n_params_d/1e6:0.04f} M")
    print(f"Number of trainable generator params:     {n_params_g/1e6:0.04f} M")
    print(f"Total trainable model params:             {(n_params_d+n_params_g)/1e6:0.04f} M")

    optimizer_dis = optim.AdamW(learning_rate=5e-5)
    optimizer_gen = optim.AdamW(learning_rate=2e-5)

    state = [model_dis.state, model_gen.state, optimizer_dis.state, optimizer_gen.state]

    fixed_noise = mx.random.normal([64, 1, 1, nz])

    generate_and_save_images(model_gen, fixed_noise, 0, save_dir)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(real_data, noise):
        loss_and_grad_fn_dis = nn.value_and_grad(model_dis, loss_fn_dis)
        loss_dis, grads_dis = loss_and_grad_fn_dis(model_dis, model_gen, real_data, noise)
        optimizer_dis.update(model_dis, grads_dis)

        loss_and_grad_fn_gen = nn.value_and_grad(model_gen, loss_fn_gen)
        loss_gen, grads_gen = loss_and_grad_fn_gen(model_dis, model_gen, noise)
        optimizer_gen.update(model_gen, grads_gen)

        return loss_dis, loss_gen

    for e in range(1, args.epochs + 1):
        train_iter = anime_faces_batch_iterator(train_images, args.batch_size)

        model_dis.train()
        model_gen.train()

        tic = time.perf_counter()
        loss_acc_dis = 0.0
        loss_acc_gen = 0.0
        throughput_acc = 0.0

        for batch_count, batch in enumerate(train_iter):
            real_data = mx.array(batch)

            noise = mx.random.normal([real_data.shape[0], 1, 1, nz])

            throughput_tic = time.perf_counter()

            loss_dis, loss_gen = step(real_data, noise)
            mx.eval(state)

            throughput_toc = time.perf_counter()
            throughput_acc += real_data.shape[0] / (throughput_toc - throughput_tic)
            loss_acc_dis += loss_dis.item()
            loss_acc_gen += loss_gen.item()

            if batch_count > 0 and (batch_count % 10 == 0):
                print(
                    " | ".join(
                        [
                            f"Epoch {e:4d}",
                            f"Dis Loss {(loss_acc_dis / batch_count):10.2f}",
                            f"Gen Loss {(loss_acc_gen / batch_count):10.2f}",
                            f"Throughput {(throughput_acc / batch_count):8.2f} im/s",
                            f"Batch {batch_count:5d}",
                        ]
                    ),
                    end="\r",
                )
        toc = time.perf_counter()

        print(
            " | ".join(
                [
                    f"Epoch {e:4d}",
                    f"Dis Loss {(loss_acc_dis / batch_count):10.2f}",
                    f"Gen Loss {(loss_acc_gen / batch_count):10.2f}",
                    f"Throughput {(throughput_acc / batch_count):8.2f} im/s",
                    f"Time {toc - tic:8.1f} (s)",
                ]
            )
        )

        model_dis.eval()
        model_gen.eval()

        generate_and_save_images(model_gen, fixed_noise, e, save_dir)

        model_dis.save_weights(str(save_dir / "weights_dis.npz"))
        model_gen.save_weights(str(save_dir / "weights_gen.npz"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU acceleration",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-13, help="Learning rate")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/gan/take_0/",
        help="Path to save the model and generated images.",
    )

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")

    main(args)