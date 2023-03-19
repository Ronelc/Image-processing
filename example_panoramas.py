import os
import sol4
import time

def main():
  is_bonus = False
  experiments = ['iguazu.mp4', 'boat.mp4']

  for experiment in experiments:
    exp_no_ext = experiment.split('.')[0]
    os.system('mkdir dump')
    os.system(('mkdir ' + str(os.path.join('dump', '%s'))) % exp_no_ext)
    os.system(('ffmpeg -i ' + str(os.path.join('videos', '%s ')) + str(os.path.join('dump', '%s', '%s%%03d.jpg'))) % (experiment, exp_no_ext, exp_no_ext))


    s = time.time()
    panorama_generator = sol4.PanoramicVideoGenerator(os.path.join('dump', '%s') % exp_no_ext, exp_no_ext, 2100,bonus=is_bonus)
    panorama_generator.align_images(translation_only='boat' in experiment)
    panorama_generator.generate_panoramic_images(9)
    print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))

    panorama_generator.save_panoramas_to_video()


if __name__ == '__main__':
  main()
