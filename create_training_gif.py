import os
import glob
import imageio
from config import NTH_SAVE, GPU


if __name__ == '__main__':
    print('Createing GIF ...')

    training_img_dir = 'training_images'
    num_training_images = len([name for name in os.listdir(training_img_dir)])
    output_name = 'MNIST_VAE_Training_{}_epochs.gif'.format(
        num_training_images)

    with imageio.get_writer(output_name, mode='I') as writer:
        filenames = glob.glob('{}/*.png'.format(training_img_dir))
        filenames = sorted(filenames)

        for i, filename in enumerate(filenames):
            # Only add every nth image or last to gif
            if i % NTH_SAVE == 0 or i == len(filenames) - 1:
                image = imageio.imread(filename)
                writer.append_data(image)

                # Add extra frames of the last image
                if i == len(filenames) - 1:
                    num_extra_frames = 5
                    for _ in range(num_extra_frames):
                        writer.append_data(image)

    print('GIF written to {}'.format(os.path.abspath(output_name)))
