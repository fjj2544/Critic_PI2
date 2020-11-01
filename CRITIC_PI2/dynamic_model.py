import tensorflow as tf
import numpy as np
# from envs.Pendulum import reward_function,is_done
# from envs.InvertedPendulum import reward_function,is_done
from envs.InvertedDoublePendulum import reward_function,is_done
# from envs.Hopper import reward_function,is_done
# from envs.Walker2d import reward_function,is_done
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
        # ------------build dataset-----
        self.build_train_model()
        # -------------- Network --------------
        with tf.variable_scope(self.name):
            # ------------------------- MLP -------------------------
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
    def build_train_model(self):
        with tf.name_scope("DataSet"):
            self.total_obs_action = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features + self.n_actions], name="input_data")  # 输入为1
            self.total_delta = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="real_output")  # 输出为1
            # create dataloader
            dataset = tf.data.Dataset.from_tensor_slices((self.total_obs_action, self.total_delta))
            dataset = dataset.shuffle(buffer_size=1000)  # choose data randomly from this buffer
            dataset = dataset.batch(32)  # batch size you will use
            dataset = dataset.repeat(None)  # 数据集遍历多少次就会停止 None就是不管
            self.iterator = dataset.make_initializable_iterator()  # later we have to initialize this one
            self.obs_action, self.delta = self.iterator.get_next()
    # Learn the model
    def learn(self, batch_obs_act, batch_dt,EPOCH=int(1e2)):
        self.sess.run(self.iterator.initializer, feed_dict={self.total_obs_action: batch_obs_act,
                                                            self.total_delta: batch_dt})

        for epoch in range(EPOCH):
            summary, _ = self.sess.run([self.summary_dynamic_loss,
                                     self.train_op])
        return summary

    def prediction(self, s_a):
        delta = self.sess.run([self.predict], feed_dict={self.obs_action: s_a})
        predict_out = delta + s_a[:, 0:self.n_features]
        predict_out = predict_out.reshape([-1, self.n_features])

        reward = np.array(reward_function(predict_out,s_a[:,self.n_features:self.n_features+self.n_actions]))
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
