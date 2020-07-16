import os
import tensorflow as tf

from model import EAST_model
from losses import dice_loss, rbox_loss
import data_processor

from absl import app
from absl import flags

flags.DEFINE_integer('input_size', default=512, help='input size for training of the network')
flags.DEFINE_integer('batch_size', default=8, help='batch size for training')
flags.DEFINE_integer('nb_workers', default=4, help='number of processes to spin up when using process based threading') # as defined in https://keras.io/models/model/#fit_generator
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate')
flags.DEFINE_float('lr_decay_rate', default=0.94, help='decay rate for the learning rate')
flags.DEFINE_integer('lr_decay_steps', default=16250, help='number of steps after which the learning rate is decayed by decay rate')
flags.DEFINE_integer('max_steps', default=100000, help='maximum number of steps')
flags.DEFINE_string('checkpoint_path', default='./east_resnet_50_rbox', help='path to a directory to save model checkpoints during training')
flags.DEFINE_integer('save_checkpoint_steps', default=50, help='period at which checkpoints are saved (defaults to every 50 steps)')
flags.DEFINE_string('training_data_path', default='./data/ICDAR2015/train_data', help='path to training data')
flags.DEFINE_integer('max_image_large_side', default=1280, help='maximum size of the large side of a training image before cropping a patch for training')
flags.DEFINE_integer('max_text_size', default=800, help='maximum size of a text instance in an image; image resized if this limit is exceeded')
flags.DEFINE_integer('min_text_size', default=10, help='minimum size of a text instance; if smaller, then it is ignored during training')
flags.DEFINE_float('min_crop_side_ratio', default=0.1, help='the minimum ratio of min(H, W), the smaller side of the image, when taking a random crop from thee input image')
flags.DEFINE_string('geometry', default='RBOX', help='geometry type to be used; only RBOX is implemented now, but the original paper also uses QUAD')
flags.DEFINE_boolean('suppress_warnings_and_error_messages', default=True, help='whether to show error messages and warnings during training (some error messages during training are expected to appear because of the way patches for training are created)')

FLAGS = flags.FLAGS

def train_step(model,
               x,
               optimizer,
               overly_small_text_region_training_mask,
               text_region_boundary_training_mask,
               small_text_weight,
               target_score_map,
               target_geo_maps,
               loss_weight):
  with tf.GradientTape() as tape:
    score_y_pred, geo_y_pred = model(x)
    _dice_loss = dice_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, loss_weight, small_text_weight, target_score_map, score_y_pred)
    _rbox_loss = rbox_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, small_text_weight, target_score_map, target_geo_maps, geo_y_pred)
    loss = _dice_loss + _rbox_loss

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def main(_):
  # check if checkpoint path exists
  if not os.path.exists(FLAGS.checkpoint_path):
    os.mkdir(FLAGS.checkpoint_path)

  train_data_generator = data_processor.generator(FLAGS)
  train_samples_count = data_processor.count_samples(FLAGS)
  print('total batches per epoch : {}'.format(train_samples_count / FLAGS.batch_size))

  east = EAST_model(FLAGS.input_size)
  east.model.summary()

  score_map_loss_weight = tf.Variable(0.01, name='score_map_loss_weight')
  small_text_weight = tf.Variable(0., name='small_text_weight')

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    FLAGS.init_learning_rate,
    decay_steps=FLAGS.lr_decay_steps,
    decay_rate=FLAGS.lr_decay_rate,
    staircase=True)

  optimizer = tf.optimizers.Adam(lr_schedule)

  # set checkpoint manager
  ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=east)
  ckpt_manager = tf.train.CheckpointManager(ckpt,
                                            directory=FLAGS.checkpoint_path,
                                            max_to_keep=5)
  latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

  # restore latest checkpoint
  if latest_ckpt:
    ckpt.restore(latest_ckpt)
    print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))

  # set tensorboard summary writer
  summary_writer = tf.summary.create_file_writer(FLAGS.checkpoint_path + '/train')

  while int(ckpt.step) < (FLAGS.max_steps + 1):
    # load data
    [input_images, overly_small_text_region_training_masks, text_region_boundary_training_masks, score_maps], \
    [target_score_maps, target_geo_maps] = next(train_data_generator)

    # update parameter
    train_step(east,
               input_images,
               optimizer,
               overly_small_text_region_training_masks,
               text_region_boundary_training_masks,
               small_text_weight,
               target_score_maps,
               target_geo_maps,
               score_map_loss_weight
               )

    score_y_pred, geo_y_pred = east(input_images)
    _dice_loss = dice_loss(overly_small_text_region_training_masks, text_region_boundary_training_masks, score_map_loss_weight,
                           small_text_weight, target_score_maps, score_y_pred)
    _rbox_loss = rbox_loss(overly_small_text_region_training_masks, text_region_boundary_training_masks,
                           small_text_weight, target_score_maps, target_geo_maps, geo_y_pred)
    loss = _dice_loss + _rbox_loss

    print('Step {:06d}, dice_loss {:.4f}, rbox_loss {:.4f}, total_loss {:.4f}'.format(int(ckpt.step), _dice_loss, _rbox_loss, loss))

    if ckpt.step % FLAGS.save_checkpoint_steps == 0:
      # save checkpoint
      ckpt_manager.save(checkpoint_number=ckpt.step)
      print('global_step : {}, checkpoint is saved!'.format(int(ckpt.step)))

      with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=int(ckpt.step))
        tf.summary.scalar('pred_score_map_loss', _dice_loss, step=int(ckpt.step))
        tf.summary.scalar('pred_geo_map_loss ', _rbox_loss, step=int(ckpt.step))
        tf.summary.scalar('learning_rate ', optimizer.lr(ckpt.step).numpy(), step=int(ckpt.step))
        tf.summary.scalar('small_text_weight', small_text_weight, step=int(ckpt.step))

        tf.summary.image("input_image", tf.cast((input_images + 1) * 127.5, tf.uint8), step=int(ckpt.step), max_outputs=3)
        tf.summary.image("overly_small_text_region_training_mask", tf.cast(overly_small_text_region_training_masks * 255, tf.uint8), step=int(ckpt.step), max_outputs=3)
        tf.summary.image("text_region_boundary_training_mask", tf.cast(text_region_boundary_training_masks * 255, tf.uint8), step=int(ckpt.step), max_outputs=3)
        tf.summary.image("score_map_target", tf.cast(target_score_maps * 255, tf.uint8), step=int(ckpt.step), max_outputs=3)
        tf.summary.image("score_map_pred", tf.cast(score_y_pred * 255, tf.uint8), step=int(ckpt.step), max_outputs=3)
        for i in range(4):
          tf.summary.image("geo_map_%d_target" % (i), tf.cast(tf.expand_dims(target_geo_maps[:, :, :, i], axis=3) / FLAGS.input_size * 255, tf.uint8), step=int(ckpt.step), max_outputs=3)
          tf.summary.image("geo_map_%d_pred" % (i), tf.cast(tf.expand_dims(geo_y_pred[:, :, :, i], axis=3) / FLAGS.input_size * 255, tf.uint8), step=int(ckpt.step), max_outputs=3)
        tf.summary.image("geo_map_4_target", tf.cast((tf.expand_dims(target_geo_maps[:, :, :, 4], axis=3) + 1) * 127.5, tf.uint8), step=int(ckpt.step), max_outputs=3)
        tf.summary.image("geo_map_4_pred", tf.cast((tf.expand_dims(geo_y_pred[:, :, :, 4], axis=3) + 1) * 127.5, tf.uint8), step=int(ckpt.step), max_outputs=3)

    ckpt.step.assign_add(1)

if __name__ == '__main__':
  app.run(main)
