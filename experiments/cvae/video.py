import mlx.core as mx
import numpy as np
import cv2
from models.cvae import CVAE  # Adjust to the correct import path

# Generate random anime_faces
def generate_random_images(model, num_images=5):
    latent_dim = model.n_latent
    # Generate random latent vectors
    z_random = mx.random.normal((num_images, latent_dim))

    # Decode latent vectors to anime_faces
    images = model.decode(z_random)

    return images

def display_image_with_opencv(image):
    # Convert image from [-1, 1] to [0, 255] and reshape for OpenCV
    image = np.array(image * 255).astype(np.uint8)

    # OpenCV expects the image in BGR format, so we convert from RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)


    # Display the image using OpenCV
    cv2.imshow('Generated Image', image)
    cv2.waitKey(1)  # Wait 1ms to create the illusion of a stream

# Main testing function
if __name__ == "__main__":
    # Load the model
    model = CVAE(64, (64, 64, 3), 256)
    model.load_weights("take_2/weights.npz")

    mx.random.seed(42)

    # Continuous image generation and display loop
    while True:
        # Generate a single random image
        images = generate_random_images(model, num_images=1)
        generated_image = images[0]

        # Display the image using OpenCV
        display_image_with_opencv(generated_image)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close OpenCV windows when done
    cv2.destroyAllWindows()
