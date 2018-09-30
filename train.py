from __future__ import print_function
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class Model():
	def __init__(self, batch_size, num_inputs, data, latent_dim = 8, encoder_hidden = 128, decoder_hidden = 128, 
		num_epochs = 10, learning_rate = 0.01, weight_decay = 0.001, data_ctx = mx.cpu(), model_ctx = mx.cpu()):
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.data_ctx = data_ctx
		self.model_ctx = model_ctx
		self.data_orig = data

		self.num_inputs = num_inputs
		self.encoder_hidden = encoder_hidden
		self.latent_dim = latent_dim
		self.decoder_hidden = decoder_hidden

		self.define_network()


	def define_network(self):
		'''
			This function defines the network of encoder and decoder
		'''
		self.encoder_net = gluon.nn.Sequential()
		self.encoder_net.add(gluon.nn.Dense(self.encoder_hidden, activation="relu"))
		self.encoder_net.add(gluon.nn.Dense(2*self.latent_dim))
		self.encoder_net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=self.model_ctx)

		self.decoder_net = gluon.nn.Sequential()
		self.decoder_net.add(gluon.nn.Dense(self.decoder_hidden, activation="relu"))
		self.decoder_net.add(gluon.nn.Dense(self.num_inputs))
		self.decoder_net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=self.model_ctx)

		self.trainer_en = gluon.Trainer(self.encoder_net.collect_params(), 'sgd', 
			{'learning_rate': self.learning_rate, 'wd': self.weight_decay})
		self.trainer_dec = gluon.Trainer(self.decoder_net.collect_params(), 'sgd', 
			{'learning_rate': self.learning_rate, 'wd': self.weight_decay})
		

	def sample_z(self, mu, sigma):   
		'''
			Latent variables
			Sample z from Normal(z|mu, sigma) using the reparameterization trick.
		'''
		loc = nd.zeros((mu.shape[0], latent_dim))
		scale = nd.ones((mu.shape[0], latent_dim))
		eps = nd.random.normal(loc,scale,dtype='float32').as_in_context(model_ctx)
		z = mu + (eps * nd.exp(sigma))
		return z

	def softmax_cross_entropy(self, yhat_linear, y):
		return (-nd.nansum(y * nd.log_softmax(yhat_linear)))

	def compute_loss(self, y, x, mu, sigma):
		'''
			Compute the total loss as sum of KL-divergence loss and cross entropy loss
			Returns: average loss per sample
		'''
		kl_term = 0.5*(mu.shape[0]*mu.shape[1] + 2.*sigma.sum() - mu.square().sum() - 2.*sigma.exp().sum())
		fitness_loss = -self.softmax_cross_entropy(y,x) # y is the output of decoder and x is the input here
		total = -(kl_term + fitness_loss)/self.batch_size
		return total

	def train(self):
		'''
			Runs the training loop in batches for a given number of epochs
		'''
		for e in range(self.num_epochs):
			cumulative_loss = 0
			num_batches = self.data_orig.shape[0]/self.batch_size
			for i in range(num_batches):
				#print(i)
				data = self.data_orig[self.batch_size*i+1:self.batch_size*(i+1)+1]
				data_norm = nd.array(normalize(data, norm='l1', axis=1).todense())
				data_norm = data_norm.as_in_context(self.data_ctx).reshape((-1, self.num_inputs))
				with autograd.record():
					encoder_output = self.encoder_net(data_norm)
					mu = encoder_output[:,:self.latent_dim]
					log_sigma = encoder_output[:,self.latent_dim:]
					latent_repr = self.sample_z(mu, log_sigma)
					decoder_output = self.decoder_net(latent_repr)
					loss = self.compute_loss(decoder_output, data_norm, mu, log_sigma)
					
				#print(loss)
				loss.backward()
				self.trainer_en.step(data.shape[0])
				self.trainer_dec.step(data.shape[0])
				cumulative_loss += nd.mean(loss).asscalar()
			print("Epoch %s, loss: %s" % (e, cumulative_loss / num_batches))


if __name__ == "__main__":

	# define the variables here
	batch_size = 256
	num_epochs = 15
	encoder_hidden = 128
	decoder_hidden = 128
	latent_dim = 8
	learning_rate = 0.1
	weight_decay = 0.001
	context = mx.gpu()
	data_ctx = context
	model_ctx = context

	#row, col, data = np.loadtxt('./docword_nytimes.txt', dtype = 'int32', skiprows = 3, unpack = True)
	#csr = csr_matrix( (data,(row,col)))
	
	csr = load_npz('nytimes_sparse.npz')
	num_inputs = csr.shape[1]

	vae = Model(batch_size = batch_size, num_epochs = num_epochs, num_inputs = num_inputs, latent_dim = latent_dim, encoder_hidden = encoder_hidden, decoder_hidden = decoder_hidden, 
		learning_rate = learning_rate, weight_decay = weight_decay, data_ctx = data_ctx, model_ctx = model_ctx, data = csr)
	vae.train()
