import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Activation, LeakyReLU, ReLU
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import array_to_img

import vutils

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', default='data/celeba', type=Path, help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=128, type=int, help='input batch size')
parser.add_argument('--image-size', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--nz', default=100, type=int, help='size of the latent z vector')
parser.add_argument('--ngf', default=64, type=int)
parser.add_argument('--ndf', default=64, type=int)
parser.add_argument('--niter', default=25, type=int, help='number of epochs to train for')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='beta1 for adam')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs to use')
parser.add_argument('--outf', default='samples', type=Path, help='folder to output images')
parser.add_argument('--log-dir', default='logs', type=Path, help='log folder to save training progresses')
parser.add_argument('--ckpt-dir', default='ckpt', help='checkpoint folder to save model checkpoints')
parser.add_argument('--manual-seed', type=int, help='manual seed')

opt = parser.parse_args()

if not opt.manual_seed:
    opt.manual_seed = random.randint(1, 10000)
print(f'Random Seed: {opt.manual_seed}')
random.seed(opt.manual_seed)
tf.random.set_seed(opt.manual_seed)

if opt.ngpu <= 0:
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
elif opt.ngpu == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
else:
    strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{i}' for i in range(opt.ngpu)])
print(f'Device type: {"CPU" if opt.ngpu <= 0 else "GPU"}')
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Number of channels in the training images. For color images this is 3
nc = 3


def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = 2 * image - 1
    shape = tf.shape(image)
    # center cropping
    h, w = shape[-3], shape[-2]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)

    image = tf.image.resize(cropped_image, [opt.image_size, opt.image_size])
    return image


# Create the dataset
dataset = tf.data.Dataset.list_files(str(opt.dataroot/'*/*')) \
    .map(parse_image, num_parallel_calls=opt.workers) \
    .batch(opt.batch_size)
dataset_dist = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    # Custom weights initialization called on netG and netD
    initializer = RandomNormal(0, 0.02)

    netG = Sequential([
        Input(shape=(1, 1, opt.nz)),
        # input is Z, going into a convolution
        Conv2DTranspose(opt.ngf * 8, 4, 1, 'valid', use_bias=False, kernel_initializer=initializer),
        BatchNormalization(),
        ReLU(),
        # state size. 4 x 4 x (ngf*16)
        Conv2DTranspose(opt.ngf * 8, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        BatchNormalization(),
        ReLU(),
        # state size. 8 x 8 x (ngf*8)
        Conv2DTranspose(opt.ngf * 4, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        BatchNormalization(),
        ReLU(),
        # state size. 16 x 16 x (ngf*2)
        Conv2DTranspose(opt.ngf * 2, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        BatchNormalization(),
        ReLU(),
        # state size. 32 x 32 x (ngf)
        Conv2DTranspose(nc, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        Activation(tf.nn.tanh, name='tanh')
        # state size. 64 x 64 x (nc)
    ], name='generator')
    netG.summary()

    netD = Sequential([
        Input(shape=(opt.image_size, opt.image_size, nc)),
        # input is 64 x 64 x (nc)
        Conv2D(opt.ndf, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        LeakyReLU(0.2),
        # state size. 32 x 32 x (ndf)
        Conv2D(opt.ndf * 2, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        BatchNormalization(),
        LeakyReLU(0.2),
        # state size. 16 x 16 x (ndf*2)
        Conv2D(opt.ndf * 4, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        BatchNormalization(),
        LeakyReLU(0.2),
        # state size. 8 x 8 x (ndf*4)
        Conv2D(opt.ndf * 8, 4, 2, 'same', use_bias=False, kernel_initializer=initializer),
        BatchNormalization(),
        LeakyReLU(0.2),
        # state size. 4 x 4 x (ndf*8)
        Conv2D(1, 4, 1, 'valid', use_bias=False, kernel_initializer=initializer),
        # state size. 1 x 1 x 1
        Reshape((1,))
    ], name='discriminator')
    netD.summary()

    # Initialize BCELoss function
    criterion = BinaryCrossentropy(from_logits=True, reduction=Reduction.NONE)


    def compute_loss(y_true, y_pred):
        per_example_loss = criterion(y_true, y_pred)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=opt.batch_size)


    # Setup Adam optimizers for both G and D
    optimizerD = Adam(learning_rate=opt.lr, beta_1=opt.beta1)
    optimizerG = Adam(learning_rate=opt.lr, beta_1=opt.beta1)

    # Setup model checkpoint
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1, trainable=False, name='epoch'),
                               step=tf.Variable(0, trainable=False, name='step'),
                               optimizerD=optimizerD,
                               optimizerG=optimizerG,
                               netD=netD,
                               netG=netG)
    ckpt_manager = tf.train.CheckpointManager(ckpt, opt.ckpt_dir, max_to_keep=None)
    ckpt_manager.restore_or_initialize()
    if ckpt_manager.latest_checkpoint:
        print(f'Restored from {ckpt_manager.latest_checkpoint}')
    else:
        print('Initializing from scratch.')

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = tf.random.normal([64, 1, 1, opt.nz])

if opt.dry_run:
    opt.niter = 1

# Set up a log directory
file_writer = tf.summary.create_file_writer(str(opt.log_dir/datetime.now().strftime('%Y%m%d-%H%M%S')))

# Set up a sample output directory
opt.outf.mkdir(parents=True, exist_ok=True)


def train_step(data):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    # Format batch
    batch_size = data.shape[0]
    # Generate batch of latent vectors
    noise = tf.random.normal([batch_size, 1, 1, opt.nz])
    # Generate fake image batch with G
    fake = netG(noise, training=True)
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass real batch through D
        real_output = netD(data, training=True)
        # Calculate loss on all-real batch
        errD_real = compute_loss(tf.ones_like(real_output), real_output)
        # Classify all fake batch with D
        fake_output = netD(fake, training=True)
        # Calculate D's loss on the all-fake batch
        errD_fake = compute_loss(tf.zeros_like(fake_output), fake_output)
    # Calculate gradients for D in backward pass
    gradients_real = tape.gradient(errD_real, netD.trainable_variables)
    gradients_fake = tape.gradient(errD_fake, netD.trainable_variables)
    # Add the gradients from the all-real and all-fake batches
    accumulated_gradients = [g1 + g2 for g1, g2 in zip(gradients_real, gradients_fake)]
    # Update D
    optimizerD.apply_gradients(zip(accumulated_gradients, netD.trainable_variables))
    D_x = tf.math.reduce_mean(tf.math.sigmoid(real_output))
    D_G_z1 = tf.math.reduce_mean(tf.math.sigmoid(fake_output))
    errD = errD_real + errD_fake

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    with tf.GradientTape() as tape:
        # Since we just updated D, perform another forward pass of all-fake batch through D
        fake = netG(noise, training=True)
        fake_output = netD(fake, training=True)
        # Calculate G's loss based on this output
        # fake labels are real for generator cost
        errG = compute_loss(tf.ones_like(fake_output), fake_output)
    # Calculate gradients for G
    gradients = tape.gradient(errG, netG.trainable_variables)
    # Update G
    optimizerG.apply_gradients(zip(gradients, netG.trainable_variables))
    D_G_z2 = tf.math.reduce_mean(tf.math.sigmoid(fake_output))

    return errD, errG, D_x, D_G_z1, D_G_z2


@tf.function
def distributed_train_step(dist_inputs):
    errD, errG, D_x, D_G_z1, D_G_z2 = strategy.run(train_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, errD, axis=None), \
           strategy.reduce(tf.distribute.ReduceOp.SUM, errG, axis=None), \
           strategy.reduce(tf.distribute.ReduceOp.MEAN, D_x, axis=None), \
           strategy.reduce(tf.distribute.ReduceOp.MEAN, D_G_z1, axis=None), \
           strategy.reduce(tf.distribute.ReduceOp.MEAN, D_G_z2, axis=None)


for epoch in range(int(ckpt.epoch.numpy()), opt.niter + 1):
    for i, data in enumerate(dataset_dist):
        errD, errG, D_x, D_G_z1, D_G_z2 = distributed_train_step(data)
        # Output training stats
        if i % 50 == 0:
            print(f'[{epoch}/{opt.niter}][{i}/{len(dataset)}]\t'
                  f'Loss_D: {errD:.4f}\t'
                  f'Loss_G: {errG:.4f}\t'
                  f'D(x): {D_x:.4f}\t'
                  f'D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
        if opt.dry_run:
            break
        # Log training stats
        ckpt.step.assign_add(1)
        step = int(ckpt.step.numpy())
        with file_writer.as_default():
            tf.summary.scalar('errD', errD, step=step)
            tf.summary.scalar('errG', errG, step=step)
            tf.summary.scalar('D_x', D_x, step=step)
            tf.summary.scalar('D_G_z1', D_G_z1, step=step)
            tf.summary.scalar('D_G_z2', D_G_z2, step=step)
    if opt.dry_run:
        break
    # Check how the generator is doing by saving G's output on fixed_noise
    fake = netG(fixed_noise, training=False)
    # Scale it back to [0, 1]
    fake = (fake + 1) / 2
    img_grid = vutils.make_grid(fake)
    with file_writer.as_default():
        tf.summary.image('Generated images', img_grid[tf.newaxis, ...], step=epoch)
    img = array_to_img(img_grid * 255, scale=False)
    img.save(opt.outf/f'fake_samples_epoch_{epoch:03d}.png')

    save_path = ckpt_manager.save()
    print(f'Saved checkpoint at epoch {epoch}: {save_path}')
    ckpt.epoch.assign_add(1)
