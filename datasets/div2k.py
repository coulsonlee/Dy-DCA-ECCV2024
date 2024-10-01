import os

import common.modes
import datasets._isr


# t means video topic and length(vlog_15)   c means chunk number and s means scale

EVAL_LR_DIR = lambda t,c,s: '/home/lee/data/{}/'.format(t) + 'DIV2K_train_LR_bicubic_{}'.format(c)+'/X{}/'.format(s)
EVAL_HR_DIR = lambda t,c,s:'/home/lee/data/{}/'.format(t) + 'DIV2K_train_HR_{}'.format(c)+'/X{}/'.format(s)
TRAIN_LR_DIR = lambda t,c,s: '/home/lee/data/{}/'.format(t) + 'DIV2K_train_LR_bicubic_{}'.format(c)+'/X{}/'.format(s)
TRAIN_HR_DIR = lambda t,c,s:'/home/lee/data/{}/'.format(t) + 'DIV2K_train_HR_{}'.format(c)+'/X{}/'.format(s)

save_file_name = []


def update_argparser(parser):
  datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--input_dir', help='Directory of input files in predict mode.')
  parser.set_defaults(
      num_channels=3,
      num_patches=1,
      train_batch_size=16,
      eval_batch_size=1,
  )


def get_dataset(mode, params):
  if mode == common.modes.PREDICT:
    return DIV2K_(mode, params)
  else:
    return DIV2K(mode, params)

def get_filename(params):
    file_name = sorted(os.listdir(TRAIN_HR_DIR(params.tt,params.chunk,params.scale)))
    file_name.sort(key=lambda x: int(x[:-4]))
    return file_name


class DIV2K(datasets._isr.ImageSuperResolutionHdf5Dataset):

  def __init__(self, mode, params):

    lr_cache_file = 'cache/div2k_{}_lr_x{}.h5'.format(mode, params.scale)
    hr_cache_file = 'cache/div2k_{}_hr.h5'.format(mode)

    lr_dir = {
        common.modes.TRAIN: TRAIN_LR_DIR(params.tt,params.chunk,params.scale),
        common.modes.EVAL: EVAL_LR_DIR(params.tt,params.chunk,params.scale),
    }[mode]
    hr_dir = {
        common.modes.TRAIN: TRAIN_HR_DIR(params.tt,params.chunk,params.scale),
        common.modes.EVAL: EVAL_HR_DIR(params.tt,params.chunk,params.scale),
    }[mode]

    lr_files = list_image_files(lr_dir)
    if mode == common.modes.PREDICT:
      hr_files = lr_files
    else:
      hr_files = list_image_files(hr_dir)

    super(DIV2K, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
        lr_cache_file,
        hr_cache_file,
    )


class DIV2K_(datasets._isr.ImageSuperResolutionDataset):

  def __init__(self, mode, params):


    lr_dir = {
        common.modes.TRAIN: TRAIN_LR_DIR(params.tt,params.chunk,params.scale),
        common.modes.EVAL: EVAL_LR_DIR(params.tt,params.chunk,params.scale),
        common.modes.PREDICT: params.input_dir,
    }[mode]
    hr_dir = {
        common.modes.TRAIN: TRAIN_HR_DIR(params.tt,params.chunk,params.scale),
        common.modes.EVAL: EVAL_HR_DIR(params.tt,params.chunk,params.scale),
        common.modes.PREDICT: '',
    }[mode]

    lr_files = list_image_files(lr_dir)
    if mode == common.modes.PREDICT:
      hr_files = lr_files
    else:
      hr_files = list_image_files(hr_dir)
    super(DIV2K_, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
    )


def list_image_files(d):
  file_name = sorted(os.listdir(d))
  file_name.sort(key=lambda x: int(x[:-4]))
  file_name = [(f, os.path.join(d, f)) for f in file_name if f.endswith('.png')]
  return file_name

