import tensorflow as tf
import numpy as np
# from tools.Pendulum import reward_function
# from tools.InvertedPendulum import reward_function,is_done
from envs.InvertedDoublePendulum import reward_function,is_done
tf.set_random_seed(1)

from tools.plot_data import mkdir
class Dynamic_Net():
    def __init__(self,
                 observation_dim,
                 action_dim,
                 name,
                 sess,
                 trainable=True,
                 lr=0.0001,
                 model_file=None):
        # -------------- Property --------------
        self.n_features = observation_dim
        self.n_actions = action_dim
        self.learning_rate = lr
        self.name = name
        self.batch = 32

        # -------------- Network --------------
        with tf.variable_scope(self.name):
            # ------------------------- MLP -------------------------
            self.obs_action = tf.placeholder(tf.float32, shape=[None, self.n_features + self.n_actions])
            self.f1 = tf.layers.dense(inputs=self.obs_action, units=120, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            self.f2 = tf.layers.dense(inputs=self.f1, units=60, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            self.f3 = tf.layers.dense(inputs=self.f2, units=60, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            self.predict = tf.layers.dense(inputs=self.f3, units=self.n_features, trainable=trainable)
        # -------------- Label --------------
        self.delta = tf.placeholder(tf.float32, [None, self.n_features])
        # -------------- Loss --------------
        self.loss = tf.reduce_mean(tf.square(self.predict - self.delta))
        self.summary_dynamic_loss = tf.summary.scalar("Dynamic Loss",self.loss)
        # -------------- Train --------------
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # -------------- Sess --------------
        try:
            self.sess = sess
        except:
            print("ERROR!")
        # -------------- Saver --------------
        self.saver = tf.train.Saver(max_to_keep=10)
        if model_file is not None:
            self.restore_model(model_file)

    # Learn the model
    def learn(self, batch_obs_act, batch_dt):
        summary, _ = self.sess.run([self.summary_dynamic_loss,
                                 self.train_op], feed_dict={self.obs_action: batch_obs_act,
                                                                       self.delta: batch_dt})
        return summary

    def prediction(self, s_a):
        delta = self.sess.run([self.predict], feed_dict={self.obs_action: s_a})
        predict_out = delta + s_a[:, 0:self.n_features]
        predict_out = predict_out.reshape([-1, self.n_features])

        reward = reward_function(predict_out,s_a[:,self.n_features:self.n_features+self.n_actions])
        done = is_done(predict_out,s_a[:,self.n_features:self.n_features+self.n_actions])


        return predict_out, reward,done
    # 定义存储模型函数
    def save_model(self, model_dir,model_name):
        mkdir(model_dir)
        model_path = model_dir+model_name
        self.saver.save(self.sess, model_path)

    # 定义恢复模型函数
    def restore_model(self, model_path='./dynamic'):
        self.saver.restore(self.sess, model_path)
        print("RESTORE COMPLETED")
