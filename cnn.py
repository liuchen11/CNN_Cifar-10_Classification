import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor.signal import downsample
from theano.tensor import shared_randomstreams

def ReLU(x):
	return theano.tensor.switch(x<0,0,x)

def Dropout_Func(rng,value,p):
	'''
	>>>type rng: numpy.random.RandomState
	>>>para rng: initalize weight randomly

	>>>type value: theano.tensor.TensorType
	>>>para value: input data

	>>>type p: float
	>>>para p: dropout rate
	'''
	srng=shared_randomstreams.RandomStreams(rng.randint(2011010539))
	mask=srng.binomial(n=1,p=1-p,size=value.shape)
	return value*T.cast(mask,theano.config.floatX)

class HiddenLayer(object):
	def __init__(self,rng,input,n_in,n_out,activation,dropout):
		'''
		>>>type rng: numpy.random.RandomState
		>>>para rng: initalize weight randomly

		>>>type input: theano.tensor.TensorType
		>>>para input: input data

		>>>type n_in: int
		>>>para n_in: the num of input neurons
		
		>>>type n_out: int
		>>>para n_out: the num of output neurons

		>>>type activation: func
		>>>para activation: the activate function

		>>>type dropout: boolean
		>>>para dropout: whether or not to use dropout
		'''
		self.input=input

		w_bound=np.sqrt(6.0/(n_in+n_out))

		w_value=np.asarray(
			rng.uniform(low=-w_bound,high=w_bound,size=(n_in,n_out)),
			dtype=theano.config.floatX
			)
		
		if activation==T.nnet.sigmoid:
			w_value*=4
		self.w=theano.shared(value=w_value,name='w',borrow=True)

		b_value=np.zeros((n_out),dtype=theano.config.floatX)
		self.b=theano.shared(value=b_value,name='b',borrow=True)

		raw_output=T.dot(input,self.w)+self.b

		self.output=(
			raw_output if activation is None
			else activation(raw_output)
			)

		# if dropout==True:
		# 	mask_vec=np.asarray(
		# 		rng.uniform(low=-10,high=10,size=(n_out)),
		# 		dtype=theano.config.floatX
		# 		)
		# 	for i in xrange(n_out):
		# 		if mask_vec[i]<0:
		# 			self.output[i]=0

		self.param=[self.w,self.b]

class DropoutHiddenLayer(HiddenLayer):
	def __init__(self,rng,input,n_in,n_out,activation,dropout_rate):
		'''
		>>>rng,input,n_in,n_out,activation is the same as above
		'''
		HiddenLayer.__init__(self,rng=rng,input=input,n_in=n_in,n_out=n_out,activation=activation,dropout=True)
		self.output=Dropout_Func(rng=rng,value=self.output,p=dropout_rate)

class LogisticRegression(object):

	def __init__(self,input,n_in,n_out):
		'''
		>>>type input: T.TensorType
		>>>para input: input data

		>>>type n_in: int
		>>>para n_in: num of input neurons

		>>>type n_out: int
		>>>para n_out: num of output neurons
		'''
		self.w=theano.shared(
			value=np.zeros((n_in,n_out),dtype=theano.config.floatX),
			name='w',
			borrow=True
			)
		self.b=theano.shared(
			value=np.zeros((n_out,),dtype=theano.config.floatX),
			name='b',
			borrow=True
			)
		self.param=[self.w,self.b]

		self.output=softmax(T.dot(input,self.w)+self.b)
		self.predict=T.argmax(self.output,axis=1)

	def negative_log_likelyhood(self,y):
		'''
		>>>calculate the negative log_likelyhood given labels of instances

		>>>type y: T.ivector
		>>>para y: right labels of instances
		'''
		return -T.mean(T.log(self.output)[T.arange(y.shape[0]),y])

	def errors(self,y):
		'''
		>>>calculate the error rate of test instances

		>>>type y: T.ivector
		>>>para y: right labels of instances
		'''
		return T.mean(T.neq(self.predict,y))


class ConvPool(object):

	def __init__(self, rng, input,shape,filters,pool,dropout):
		'''
		>>>type rng: numpy.random.RandomState
		>>>para rng: initalize weight randomly

		>>>type input: T.dtensor4
		>>>para input: image

		>>>type shape: tuple or list of length 4
		>>>para shape: (batch size, num of input feature maps, image height, image width)

		>>>type filters: tuple or list of length 4
		>>>para filters: (num of filters, num of input feature maps, filter height, filter width)

		>>>type pool: tuple or list of length 2
		>>>para pool: pooling size

		>>>type dropout: boolean
		>>>para dropout: whether or not to use dropout
		'''

		assert filters[1]==shape[1]
		self.input=input

		#num of input to each hidden unit
		inflow=np.prod(filters[1:])

		#num of gradients from the upper layer
		outflow=filters[0]*np.prod(filters[2:])/np.prod(pool)

		w_bound=np.sqrt(6./(inflow+outflow))

		self.w=theano.shared(
			np.asarray(
				rng.uniform(low=-w_bound,high=w_bound,size=filters),
				dtype=theano.config.floatX
				),
			borrow=True
			)

		#bias
		self.b=theano.shared(
			value=np.zeros((filters[0]),dtype=theano.config.floatX),
			borrow=True
			)

		#build up convolutional layer
		conv_out=conv.conv2d(
			input=input,
			filters=self.w,
			filter_shape=filters,
			image_shape=shape
			)

		#build up pooling layer
		pool_out=downsample.max_pool_2d(
			input=conv_out,
			ds=pool,
			ignore_border=True
			)

		self.output=ReLU(pool_out+self.b.dimshuffle('x',0,'x','x'))
		# if dropout==True:
		# 	shape=self.output.shape
		# 	self.output=self.output.flatten(1)
		# 	mask_vec=np.asarray(
		# 		rng.uniform(low=-10,high=10,size=(self.output.shape[0])),
		# 		dtype=theano.config.floatX
		# 		)
		# 	for i in xrange(self.output.shape[0]):
		# 		if mask_vec[i]<0:
		# 			self.output[i]=0
		# 	self.output=self.output.reshape(shape)
		self.param=[self.w,self.b]

class DropoutConvPool(ConvPool):
	def __init__(self,rng,input,shape,filters,pool,dropout_rate):
		ConvPool.__init__(self,rng=rng,input=input,shape=shape,filters=filters,pool=pool,dropout=True)
		self.output=Dropout_Func(rng=rng,value=self.output,p=dropout_rate)

class model(object):

	def __init__(self,learn_rate,n_epochs,filters,batch_size):
		'''
		>>>type learn_rate: float
		>>>para learn_rate: learning rate used for SGD

		>>>type n_epochs: int
		>>>para n_epochs: max num of epochs to return the optimizer

		>>>type filters: tuple or list
		>>>para filters: num of filters in each layer

		>>>type batch_size: int
		>>>para batch_size: size of a batch
		'''

		rng=np.random.RandomState(20150119)
		self.learn_rate=learn_rate
		self.n_epochs=n_epochs
		self.batch_size=batch_size
		self.filters=filters

		self.x=T.dmatrix('x')		#inputs
		self.y=T.lvector('y')		#labels
		self.iter=T.iscalar('iter')	#iter_num
		self.lr=T.dscalar('lr')		#learning rate

		print 'building the model'

		input=self.x.reshape((batch_size,3,32,32))

		self.layer0=DropoutConvPool(
			rng,
			input=input,
			shape=[batch_size,3,32,32],
			filters=[filters[0],3,3,3],
			pool=[1,1],
			dropout_rate=0
			)

		self.layer1=DropoutConvPool(
			rng,
			input=self.layer0.output,
			shape=[batch_size,filters[0],30,30],
			filters=[filters[1],filters[0],3,3],
			pool=[2,2],
			dropout_rate=0
			)

		self.layer2=DropoutConvPool(
			rng,
			input=self.layer1.output,
			shape=[batch_size,filters[1],14,14],
			filters=[filters[2],filters[1],5,5],
			pool=[1,1],
			dropout_rate=0
			)

		self.layer3=DropoutConvPool(
			rng,
			input=self.layer2.output,
			shape=[batch_size,filters[2],10,10],
			filters=[filters[3],filters[2],5,5],
			pool=[2,2],
			dropout_rate=0
			)

		self.layer4=DropoutHiddenLayer(
			rng,
			input=self.layer3.output.flatten(2),
			n_in=filters[3]*3*3,
			n_out=400,
			activation=ReLU,
			dropout_rate=0
			)

		self.layer5=DropoutHiddenLayer(
			rng,
			input=self.layer4.output,
			n_in=400,
			n_out=100,
			activation=T.tanh,
			dropout_rate=0
			)

		self.layer6=LogisticRegression(
			input=self.layer5.output,
			n_in=100,
			n_out=10
			)

		self.cost=self.layer6.negative_log_likelyhood(self.y)
		self.error=self.layer6.errors(self.y)

		self.debug=self.cost

		self.params=self.layer6.param+self.layer5.param+self.layer4.param+self.layer3.param+self.layer2.param+self.layer1.param+self.layer0.param
		self.grads=T.grad(self.cost,self.params)

		self.updates=[
			(param_i,param_i-self.lr*grad_i)
			for param_i, grad_i in zip(self.params,self.grads)
		]
		print 'construction completed!'

	def train_validate_test(self,train_set_x,train_set_y,validate_set_x,validate_set_y,test_set_x,test_set_y):
		'''
		>>>add validate set to avoid overfitting

		>>>type train_set_x/validate_set_x/test_set_x: T.dmatrix
		>>>para train_set_x/validate_set_x/test_set_x: data of instances of training/validate/test set

		>>>type train_set_y/validate_set_y/test_set_y: T.ivector
		>>>para train_set_y/validate_set_y/test_set_y: labels of instances of training/validate/test set
		'''
		assert train_set_x.shape[0]==train_set_y.shape[0]
		assert validate_set_x.shape[0]==validate_set_y.shape[0]
		assert test_set_x.shape[0]==test_set_y.shape[0]

		trainX=theano.shared(train_set_x,borrow=True)
		trainY=theano.shared(train_set_y,borrow=True)
		validateX=theano.shared(validate_set_x,borrow=True)
		validateY=theano.shared(validate_set_y,borrow=True)
		testX=theano.shared(test_set_x,borrow=True)
		testY=theano.shared(test_set_y,borrow=True)

		train_set_batches=train_set_y.shape[0]/self.batch_size
		validate_set_batches=validate_set_y.shape[0]/self.batch_size
		test_set_batches=test_set_y.shape[0]/self.batch_size

		index=T.iscalar('index')
		l_r=T.dscalar('lr')
		epoch_num=T.iscalar('epoch_num')

		train_model=theano.function(
			[index,l_r],
			self.cost,
			updates=self.updates,
			givens={
			self.x:trainX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:trainY[index*self.batch_size:(index+1)*self.batch_size],
			self.lr:l_r
			}
			)
		validate_model=theano.function(
			[index],
			self.error,
			givens={
			self.x:validateX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:validateY[index*self.batch_size:(index+1)*self.batch_size]
			}
			)
		test_model=theano.function(
			[index],
			1.0-self.error,
			givens={
			self.x:testX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:testY[index*self.batch_size:(index+1)*self.batch_size]
			}
			)
		test_train=theano.function(
			[index],
			1.0-self.error,
			givens={
			self.x:trainX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:trainY[index*self.batch_size:(index+1)*self.batch_size]
			}
			)

		max_iter=10000
		max_iter_increase=2
		improvement_threshold=1.0
		validate_fre=min(train_set_batches,max_iter/2)

		best_validation_loss=np.inf
		best_iter=0
		test_score_mean=0.0

		epoch=0
		done_looping=False

		rate=self.learn_rate
		min_error=1.0
		present_error=1.0

		while (epoch<self.n_epochs) and (not done_looping):
			epoch+=1
			if present_error>0.5:
				rate=0.01
			elif present_error>0.35:
				rate=0.005
			elif present_error>0.3:
				rate=0.003
			elif present_error>0.25:
				rate=0.002
			else:
				rate=0.001

			for batch_index in xrange(train_set_batches):
				iter_num=(epoch-1)*train_set_batches+batch_index
				#print self.layer0.w.get_value()[0,0,0,0]
				if iter_num%100==0:
					print 'training@iter=%d/%d'%(iter_num,train_set_batches*self.n_epochs)
#					train_test_losses=[
#					test_train(i)
#					for i in xrange(train_set_batches)
#					]
#					train_test_mean=np.mean(train_test_losses)
#					print 'train accuracy %f %%'%train_test_mean
				
				cost_now=train_model(batch_index,rate)

				if (iter_num+1)%validate_fre==0:
					validation_losses=[
					validate_model(i)
					for i in xrange(validate_set_batches)
					]
					validation_loss_mean=np.mean(validation_losses)
					print 'epoch %i, batch_index %i/%i, validation accuracy %f %%'%(epoch,batch_index+1,train_set_batches,(1.0-validation_loss_mean)*100.)
					print 'best result:%f %%'%((1.0-min_error)*100)

					if validation_loss_mean<best_validation_loss:
						if validation_loss_mean<best_validation_loss*improvement_threshold:
							max_iter=max(max_iter,iter_num*max_iter_increase)

						best_validation_loss=validation_loss_mean
						best_iter=iter_num
						stop_optimal=0

						test_scores=[
						test_model(i)
						for i in xrange(test_set_batches)
						]
						test_score_mean=np.mean(test_scores)
						print '\nepoch %i, batch_index %i/%i, test accuracy %f %%'%(epoch,batch_index+1,train_set_batches,test_score_mean*100.)

						present_error=1-test_score_mean
						if present_error<min_error:
							min_error=present_error
						

				if iter_num>=max_iter:
					done_looping=True
					break

		print 'best validate accuracy of %f %% at iteration %i, with test accuracy %f %%'%((1.0-best_validation_loss)*100.,best_iter+1,test_score_mean*100.)
