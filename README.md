# Deep Convolutional Generative Adversarial Networks

[PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)의 Tensorflow implementation 및 한국어 번역.

데이터셋은 Celeb-A Faces dataset 사용.

## Usage

```plaintext
usage: train.py [-h] [--dataroot DATAROOT] [--workers WORKERS]
                [--batch-size BATCH_SIZE] [--image-size IMAGE_SIZE] [--nz NZ]
                [--ngf NGF] [--ndf NDF] [--niter NITER] [--lr LR]
                [--beta1 BETA1] [--dry-run] [--ngpu NGPU] [--outf OUTF]
                [--log-dir LOG_DIR] [--ckpt-dir CKPT_DIR]
                [--manual-seed MANUAL_SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to dataset (default: data/celeba)
  --workers WORKERS     number of data loading workers (default: 2)
  --batch-size BATCH_SIZE
                        input batch size (default: 128)
  --image-size IMAGE_SIZE
                        the height / width of the input image to network
                        (default: 64)
  --nz NZ               size of the latent z vector (default: 100)
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for (default: 25)
  --lr LR               learning rate (default: 0.0002)
  --beta1 BETA1         beta1 for adam (default: 0.5)
  --dry-run             check a single training cycle works (default: False)
  --ngpu NGPU           number of GPUs to use (default: 1)
  --outf OUTF           folder to output images (default: samples)
  --log-dir LOG_DIR     log folder to save training progresses (default: logs)
  --ckpt-dir CKPT_DIR   checkpoint folder to save model checkpoints (default:
                        ckpt)
  --manual-seed MANUAL_SEED
                        manual seed (default: None)

```