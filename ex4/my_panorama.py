import os
import sol4
import time


def main():
    experiment = "my_panorama.mp4"
    exp_no_ext = experiment.split('.')[0]
    os.system('mkdir dump')
    os.system(('mkdir ' + str(os.path.join('dump', '%s'))) % exp_no_ext)
    os.system(('ffmpeg -i ' + str(os.path.join('videos', '%s ')) + str(
        os.path.join('dump', '%s', '%s%%03d.jpg'))) % (
                  experiment, exp_no_ext, exp_no_ext))
    panorama_generator = sol4.PanoramicVideoGenerator(
        os.path.join('dump', '%s') % exp_no_ext, exp_no_ext, 2100, bonus=False)
    panorama_generator.align_images(translation_only=True)
    panorama_generator.generate_panoramic_images(9)
    panorama_generator.save_panoramas_to_video()


if __name__ == '__main__':
    main()
