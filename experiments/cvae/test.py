import mlx.core as mx
import matplotlib.pyplot as plt
from models.cvae import CVAE  # Adjust to the correct import path

# Generate random anime_faces
def generate_random_images(model, num_images=5):
    latent_dim = model.n_latent
    # Generate random latent vectors
    z_random = mx.random.normal((num_images, latent_dim))

    # Decode latent vectors to anime_faces
    images = model.decode(z_random)

    return images

def display_images(images):
    num_images = len(images)
    cols = min(num_images, 5)  # Display 5 anime_faces per row
    rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=(cols * 2, rows * 2))  # Adjust the figure size accordingly

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])  # Adjust if needed for image shape
        plt.axis('off')

    plt.show()



# Main testing function
if __name__ == "__main__":
    # Load the model
    model = CVAE(64, (64, 64, 3), 256)
    model.load_weights("take_2/weights.npz")

    mx.random.seed(42)

    # Generate random anime_faces
    images = generate_random_images(model, num_images=25)

    # Display the generated anime_faces
    display_images(images)
