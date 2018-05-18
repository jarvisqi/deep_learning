import os
import datetime
import numpy as np

from deblur_gan import Gan
from keras.optimizers import Adam

n_images=1,
batch_size=16,
epoch_num=4,
critic_updates = 5


def load_images(dir, num):
    """加载图片数据
    
    Arguments:
        dir {[type]} -- [description]
        num {[type]} -- [description]
    """

    pass


def save_all_weights(d, g, epoch_number, current_loss):
    """保存权重
    
    Arguments:
        d {[type]} -- [description]
        g {[type]} -- [description]
        epoch_number {[type]} -- [description]
        current_loss {[type]} -- [description]
    """

    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs():
    """训练
    """

    data = load_images('./images/train', n_images)
    x_train, y_train = data["A"], data["B"]

    g = Gan.generator_model()
    d = Gan.discriminator_model()
    d_on_g = Gan.generator_containing_discriminator_multiple(g, d)

    d_opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=Gan.wasserstein_loss)
    d.trainable = False
    loss = [Gan.perceptual_loss, Gan.wasserstein_loss]
    loss_weight = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss,  loss_weights=loss_weight)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.zeros((batch_size, 1))

    for epoch in range(epoch_num):
        print('epoch: {}/{}'.format(epoch, epoch_num))
        print('batches: {}'.format(x_train.shape[0] / batch_size))

        permutated_indexes = np.random.permutation(x_train.shape[0])
        d_loss, d_on_g_loss = [], []
        for index in range(int(x_train[0]/batch_size)):
            batch_indexes = permutated_indexes[index * batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch,batch_size=batch_size)

            for _ in range(credits):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            print("batch {} d_loss : {}".format(index+1, np.mean(d_loss)))

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_loss.append(d_on_g_loss)
            print('batch {} d_on_g_loss : {}'.format(index+1, d_on_g_loss))

            d.trainable = True
        
        # 写日志
        with open("./logs/log.txt","a") as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        # 保存权重
        save_all_weights(d,g,epoch, int(np.mean(d_on_g_losses)))
