import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow.keras as keras
from Model import DeepLabV3Plus
from DataLoader import DataLoader

model_cls = DeepLabV3Plus(256, 12)
model = model_cls.build()

root = 'D:/sideproject/cmavid'
Loader = DataLoader(root=root)

trainCnt = 369
valCnt = 100

train_ds = Loader.mk_ds(type='train')
val_ds = Loader.mk_ds(type='val')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
loss_mean = keras.metrics.Mean()
train_miou = keras.metrics.MeanIoU(num_classes=12)
val_miou = keras.metrics.MeanIoU(num_classes=12)
# set tensorboard
train_log_dir = 'logs/DeepLabv3Plus/camvid/first'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# set checkpoint
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
ckp_path = 'D:/sideproject/ckp/DeepLabv3Plus/cmavid/first'
manager = tf.train.CheckpointManager(ckpt, ckp_path, max_to_keep=None)

""" Training Loop """

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, trainable=True)
        loss_item = loss_fn(y,logits)

    grads = tape.gradient(loss_item, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_miou.update_state(y, logits)
    loss_mean.update_state(loss_item)

    return loss_item


@tf.function
def val_step(x, y):
    logits = model(x, trainable=False)
    val_miou.update_state(y, logits)


epochs = 100
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")

    train_ds_shuffle = Loader.configure_for_performance(train_ds, trainCnt, shuffle=True)
    train_iter = iter(train_ds_shuffle)

    val_ds_shuffle = Loader.configure_for_performance(val_ds, valCnt, shuffle=False)
    val_iter = iter(val_ds_shuffle)

    for step in range(trainCnt):
        img, mask = next(train_iter)
        loss_item = train_step(img, mask)

    result_loss = loss_mean.result()
    result_train_miou = train_miou.result()
    save_path = manager.save()

    print(f'Training [ Loss : {result_loss},   Miou : {result_train_miou} ]')

    for step in range(valCnt):
        img, mask = next(val_iter)
        val_loss_item = val_step(img, mask)

    result_val_miou = val_miou.result()

    print(f'Validation [ Miou : {result_val_miou} ]')
    train_miou.reset_state()
    loss_mean.reset_state()
    val_miou.reset_state()

    with train_summary_writer.as_default():
        tf.summary.scalar('Training Miou', result_train_miou, step=epoch)
        tf.summary.scalar('Loss', result_loss, step=epoch)
        tf.summary.scalar('Validation Miou', result_val_miou, step=epoch)


