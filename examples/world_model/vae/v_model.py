"""

addapted from [1]: https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py

comments:

- The KL Loss is defined slightly different in [1]:
```
# augmented kl loss per dim
self.kl_loss = - 0.5 * tf.reduce_sum(
    (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
     reduction_indices = 1
)

self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
self.kl_loss = tf.reduce_mean(self.kl_loss)

self.loss = self.r_loss + self.kl_loss
```

"""
from gempy.torch.encoder import ConvEncoder
from gempy.torch.decoder import ConvTDecoder
from gempy.torch.variational_auto_encoder import VariationalAutoEncoder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from torch import cuda


MODEL_PATH = 'examples/models/world_model/'
DEFAULT_LATENT_DIM = 32
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-4


class VModel(VariationalAutoEncoder):
    def __init__(self,
                 latent_dim=DEFAULT_LATENT_DIM,
                 encoder=None,
                 decoder=None,
                 batch_size=DEFAULT_BATCH_SIZE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 model_path=MODEL_PATH,
                 **kwargs
                 ):
        if encoder is None:
            encoder = self._build_encoder(latent_dim)

        if decoder is None:
            assert isinstance(encoder, ConvEncoder)
            latent_upscale = encoder.conv_stack_shape_out
            # latent_upscale = (1, 1, product(latent_upscale))
            decoder = self._build_decoder(latent_dim, latent_upscale)

        self.train_dataloader = None
        self.val_dataloader = None
        self.model_path = model_path

        super(VModel, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size,
                                     **kwargs)

        self.save_hyperparameters()

    @staticmethod
    def _build_encoder(latent_dim) -> ConvEncoder:
        return ConvEncoder(
            input_shape=[3, 64, 64],
            filters=[32, 64, 128, 256],
            kernels_size=[4, 4, 4, 4],
            strides=[2, 2, 2, 2],
            activation='relu',
            latent_dim=(latent_dim, latent_dim),
            latent_labels=('mu', 'var_log'),
            use_dropout=0.25,
            use_batch_norm=True,
            padding_mode='replicate',
        )

    def _build_decoder(self, latent_dim, latent_upscale) -> ConvTDecoder:
        return ConvTDecoder(
            latent_dim=latent_dim,
            latent_upscale=latent_upscale,
            filters=[128, 64, 32, 3],
            kernels_size=[4, 4, 4, 4],
            strides=[2, 2, 2, 2],
            activation=['relu', 'relu', 'relu', 'sigmoid'],
            latent_labels=('mu', 'var_log'),
            use_dropout=0.25,
            use_batch_norm=True,
            padding_mode='zeros'
        )

    @property
    def accelerator(self):
        device = 'cpu'
        if cuda.is_available():
            print(f'GPU available')
            device = 'cuda'

        return device

    def set_train_dataloader(self, value):
        self.train_dataloader = value

    def set_val_dataloader(self, value):
        self._val_dataloader = value

    def get_tune_kwargs(self, tune=True):
        if not tune:
            return {}

        return dict(
            auto_scale_batch_size='binsearch',  # run batch size scaling, result overrides hparams.batch_size
            auto_lr_find=True,                  # run learning rate finder, results override hparams.learning_rate
        )

    def fit(self, train_dataloader=None, model_path=None, val_dataloader=None, max_epochs=10, tune=False):

        # Init ModelCheckpoint callback, monitoring 'val_loss'
        print('initialize checkpoints')
        checkpoint_callback_loss = ModelCheckpoint(monitor='loss', save_top_k=1, mode='min')
        checkpoint_callback_val_loss = ModelCheckpoint(monitor='val_loss')
        checkpoint_callback_r_loss = ModelCheckpoint(monitor='r_loss')
        checkpoint_callback_kl_loss = ModelCheckpoint(monitor='KL_loss')

        if train_dataloader is not None:
            self.set_train_dataloader(train_dataloader)
            train_dataloader = self.train_dataloader

        if val_dataloader is None:
            self.set_val_dataloader(val_dataloader)
            val_dataloader = self.val_dataloader

        tune_kwargs = self.get_tune_kwargs(tune)

        print('initialize trainer')
        trainer = Trainer(max_epochs=max_epochs,
                          gpus=int('cuda' in self.accelerator),
                          default_root_dir=self.model_path if model_path is None else model_path,
                          callbacks=[checkpoint_callback_loss,
                                     checkpoint_callback_val_loss,
                                     checkpoint_callback_r_loss,
                                     checkpoint_callback_kl_loss,
                                     ],
                          precision=16,  # accelerate
                          check_val_every_n_epoch=1,
                          **tune_kwargs
                          )

        if tune:
            print('tune initial hyper-parameters:', list(tune_kwargs.keys()))
            trainer.tune(self)

        print('start training')
        trainer.fit(self, train_dataloader, val_dataloader)

        print('done')
        return self


if __name__ == '__main__':
    vmodel = VModel()

    print('Encoder:')
    print('- label: {:>20s}'.format('input'), '\tshape:', vmodel.encoder.conv_stack_shape_in)
    for l in vmodel.encoder.conv_stack:
        print('- label: {:>20s}'.format(l[0]), '\tshape:', l[-1], '\tactivation:', l[-2])
    for l in vmodel.encoder.latent_stack:
        print('- label: {:>20s}'.format(l[0]), '\tshape:', l[-1], '\tactivation:', l[-2])

    print()
    print('Decoder:')
    for l in vmodel.decoder.latent_stack:
        print('- label: {:>20s}'.format(l[0]), '\tshape:', l[-1], '\tactivation:', l[-2])
    for l in vmodel.decoder.deconv_stack:
        print('- label: {:>20s}'.format(l[0]), '\tshape:', l[-1], '\tactivation:', l[-2])
