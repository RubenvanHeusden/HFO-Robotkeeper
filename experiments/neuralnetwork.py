import tensorflow as tf

class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, num_layers, layer_sizes, learning_rate):
        assert num_layers == len(layer_sizes)
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = []
        self.learning_rate = learning_rate
        
        self.input_layer = tf.placeholder(shape=[None, self.num_inputs],
                           dtype=tf.float32, name="input_layer")
        
        last = self.input_layer
        for x in range(num_layers):
            with tf.name_scope("hidden_layer"+str(x)):
                self.layers.append(tf.contrib.layers.fully_connected(last,
                                            num_outputs = layer_sizes[x],
                                            weights_initializer=tf.contrib.layers.xavier_initializer()))
            last = self.layers[x]
        
                                        

        with tf.name_scope("fully_connected_output_layer"):   
            self.output_layer = tf.contrib.layers.fully_connected(self.layers[-1],
                    num_outputs = num_outputs,  
                    weights_initializer = tf.contrib.layers.xavier_initializer(),activation_fn=None)

        with tf.name_scope("prediction"):
            self.predicted_q_val = tf.argmax(self.output_layer, 1)
        
        self.target_q = tf.placeholder(shape=[None, num_outputs], 
        dtype=tf.float32, name="target")
        
        with tf.name_scope("loss_function"):
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target_q, 
                                                        self.output_layer)))
            
        with tf.name_scope("train_step"):
                self.update_model = tf.train.AdamOptimizer(self.learning_rate).\
                minimize(self.loss)
               
