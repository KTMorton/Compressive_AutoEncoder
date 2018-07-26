import tensorflow as tf
import numpy as np
import random
import copy
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class AutoEncoder():
    def __init__(self, learningRate, trainingSteps):
        self.learningRate = learningRate
        self.trainingSteps = trainingSteps
        self.X = 0
        self.Y = 0
        self.keep_prob = 1
        self.e_logits = 0
        self.weights = {}
        self.biases = {}
        self.train_op = 0
        self.loss_op = 0
        self.accuracy = 0
        self.d_logits = 0

    def encoder(self, x):
        encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_w1']), self.biases['encoder_b1']))
        encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, self.weights['encoder_w2']), self.biases['encoder_b2']))
        drop_2 = tf.nn.dropout(encoder_layer_2, self.keep_prob)
        compressed_layer = tf.nn.sigmoid(tf.add(tf.matmul(drop_2, self.weights['e_compressed']), self.biases['e_compressed']))
        return compressed_layer

    def decoder(self, compressed_representation):
        decoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(compressed_representation, self.weights['d_compressed']), self.biases['d_compressed']))
        decoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer_1, self.weights['decoder_w2']), self.biases['decoder_b2']))
        drop_3 = tf.nn.dropout(decoder_layer_2, self.keep_prob)
        out = tf.nn.sigmoid(tf.add(tf.matmul(drop_3, self.weights['decoder_w1']), self.biases['decoder_b1']))

        return out

    def build(self, input_size, hidden_layer1_size, hidden_layer2_size, compressed_layer_size):

        self.X = tf.placeholder("float", [None, input_size])
        self.Y = tf.placeholder("float", [None, input_size])
        self.keep_prob = tf.placeholder(tf.float32)

        self.weights = {
            'encoder_w1': tf.Variable(tf.random_normal([input_size, hidden_layer1_size])),
            'encoder_w2': tf.Variable(tf.random_normal([hidden_layer1_size, hidden_layer2_size])),
            'e_compressed': tf.Variable(tf.random_normal([hidden_layer2_size, compressed_layer_size])),
            'd_compressed': tf.Variable(tf.random_normal([compressed_layer_size, hidden_layer2_size])),
            'decoder_w2': tf.Variable(tf.random_normal([hidden_layer2_size, hidden_layer1_size])),
            'decoder_w1': tf.Variable(tf.random_normal([hidden_layer1_size, input_size]))
        }

        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([hidden_layer1_size])),
            'encoder_b2': tf.Variable(tf.random_normal([hidden_layer2_size])),
            'e_compressed': tf.Variable(tf.random_normal([compressed_layer_size])),
            'd_compressed': tf.Variable(tf.random_normal([hidden_layer2_size])),
            'decoder_b2': tf.Variable(tf.random_normal([hidden_layer1_size])),
            'decoder_b1': tf.Variable(tf.random_normal([input_size])),
        }

        self.e_logits = self.encoder(self.X)
        self.d_logits = self.decoder(self.e_logits)

        loss = tf.pow(self.Y - self.d_logits, 2)
        reg = (0.01 * (tf.nn.l2_loss(self.weights['encoder_w1']) + tf.nn.l2_loss(self.weights['encoder_w2']) + tf.nn.l2_loss(self.weights['e_compressed']) + tf.nn.l2_loss(self.weights['d_compressed']) + tf.nn.l2_loss(self.weights['decoder_w2']) + tf.nn.l2_loss(self.weights['decoder_w1'])))

        self.loss_op = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.train_op = optimizer.minimize(self.loss_op)

        # delta = tf.norm(self.X-d_logits, ord='euclidean')
        #
        # # correct_pred = tf.less(delta, 200)
        # # self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #


    def get_batch(self, size, list_of_input):
        batch_x = []
        for i in range(size):
            x = random.choice(list_of_input)
            batch_x.append(x)
        return batch_x

    def test(self, saved_file, state, input1):
        saver = tf.train.Saver()
        with tf.Session() as sess1:
            # Restore variables from disk.
            saver.restore(sess1, saved_file)
            print("Model restored.")

            output = sess1.run(self.d_logits, feed_dict={self.X: np.array(input1).reshape(1, 784), self.keep_prob: 1.0})

            return output, sess1.run(self.e_logits, feed_dict={self.X: np.array(input1).reshape(1, 784), self.keep_prob: 1.0})




    def train(self, batch_size, display_step):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            n = 1
            canvas_orig = np.empty((28 * n, 28 * n))
            canvas_recon = np.empty((28 * n, 28 * n))

            sess.run(init)

            for step in range(1, self.trainingSteps+1):

                batch_x, _ = mnist.train.next_batch(batch_size)
                batch_y = copy.deepcopy(batch_x)

                if step % 10 == 0:

                    batch_x = sess.run(tf.add(batch_x, tf.random_normal(shape=[784], mean=0.005, stddev=0.035)))

                    # for i in range(n):
                    #     # MNIST test set
                    #     # Encode and decode the digit image
                    #     output_image = batch_x
                    #
                    #     # Display reconstructed images
                    #     for j in range(n):
                    #         # Draw the reconstructed digits
                    #         canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    #             output_image[j].reshape([28, 28])
                    #
                    #     print("Reconstructed Images")
                    #     plt.figure(figsize=(n, n))
                    #     plt.imshow(canvas_recon, origin="upper", cmap="gray")
                    #     plt.show()






                #print(np.array(batch_x[0]), np.array(batch_y[0]))

                sess.run(self.train_op, feed_dict={self.X: np.array(batch_x), self.Y: np.array(batch_y), self.keep_prob: 0.85})

                if step % display_step == 0 or step == 1:

                    loss = sess.run(self.loss_op, feed_dict={self.X: np.array(batch_x), self.Y: np.array(batch_y), self.keep_prob: 0.85})
                    print("Step " + str(step) + ", Minibatch Loss= " +
                          "{:.4f}".format(loss))

                if step % 2000 == 0:


                    for i in range(n):
                        # MNIST test set
                        batch_x, _ = mnist.test.next_batch(n)
                        # Encode and decode the digit image
                        output_image = sess.run(self.d_logits, feed_dict={self.X: np.array(batch_x), self.keep_prob: 1.0})

                        # Display reconstructed images
                        for j in range(n):
                            # Draw the reconstructed digits
                            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                                output_image[j].reshape([28, 28])

                        print("Reconstructed Images")
                        plt.figure(figsize=(n, n))
                        plt.imshow(canvas_recon, origin="upper", cmap="gray")
                        plt.show()
                # if loss < 0.058:
                #     break

                    #print("Testing Accuracy: ", sess.run(self.accuracy, feed_dict={self.X: np.array(test_image_vector_list)}))
            save_path = saver.save(sess, "/home/student/PycharmProjects/KairoM/autoencoder_model_sig_drop_22.ckpt")

            print("Optimization Finished!")

# training_image_vector_list = []
# testing_image_vector_list = []

image_width, image_height, color_channels = 28,28,1
#
#
# for image_vec in load_training_data():
#     training_image_vector_list.append(image_vec.reshape(image_height*image_height*color_channels,))
# for image_vec in load_test_data():
#     testing_image_vector_list.append(image_vec.reshape(image_height*image_height*color_channels,))
autoencoder1 = AutoEncoder(0.001, 20000)

autoencoder1.build((image_height*image_height*color_channels), 261, 87, 22) # Best network so far has layers of 314, 125, 50
#autoencoder1.train(128, 500)

# autoencoder1.train(300, 500)

# autoencoder2 = AutoEncoder(0.001, 20000)
# autoencoder2.build((image_height*image_height*color_channels), 314, 125, 50)



decoded_data, encoded_data = autoencoder1.test("/home/student/PycharmProjects/KairoM/autoencoder_model_sig_drop_22.ckpt", 'encode', mnist.test.images[0])

np.save("./compressed_data/compress_22_0.txt", encoded_data)



# new_image1_flatten_array = []
# old_image_flatten_array = []
# new_image2_flatten_array = []
#
# for array in decoded_data:
#         for pixel in array:
#             new_image1_flatten_array.append(pixel*255)
#
#     # for array in decoded_data1:
#     #     for pixel1 in array:
#     #         new_image2_flatten_array.append(pixel1*255)
#     # for array in mnist.test.images[idx]:
#     #     old_image_flatten_array.append(array*255)
#
#     #print(new_image_flatten_array)
#     #old_image = np.array(old_image_flatten_array).reshape(28, 28)
# new_image1 = np.array(new_image1_flatten_array).reshape(28, 28)
#     # new_image2 = np.array(new_image2_flatten_array).reshape(28, 28)
#
# img1 = Image.fromarray(new_image1)
# img1 = img1.convert('RGB')
#  img1.save("./compressed_img/" + str(np.argmax(mnist.test.labels[idx])) + "/_22_" + str(idx) + ".jpg")
    # img2 = Image.fromarray(new_image2)
    # img2.save("./compressed_img/" + str(mnist.test.labels[idx]) + "/_50.png")
    # img2.show()






