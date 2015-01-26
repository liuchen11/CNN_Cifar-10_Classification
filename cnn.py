import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor.signal import downsample

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

		self.output=T.tanh(pool_out+self.b.dimshuffle('x',0,'x','x'))
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

		self.x=T.dmatrix('x')#inputs
		self.y=T.lvector('y')#labels
		self.iter=T.iscalar('iter')#iter_num

		print 'building the model'

		input=self.x.reshape((batch_size,3,32,32))

		self.layer0=ConvPool(
			rng,
			input=input,
			shape=[batch_size,3,32,32],
			filters=[filters[0],3,5,5],
			pool=[2,2],
			dropout=True
			)

		self.layer1=ConvPool(
			rng,
			input=self.layer0.output,
			shape=[batch_size,filters[0],14,14],
			filters=[filters[1],filters[0],5,5],
			pool=[2,2],
			dropout=True
			)

		self.layer2=HiddenLayer(
			rng,
			input=self.layer1.output.flatten(2),
			n_in=filters[1]*5*5,
			n_out=500,
			activation=T.tanh,
			dropout=True
			)

		self.layer3=LogisticRegression(
			input=self.layer2.output,
			n_in=500,
			n_out=10
			)

		self.cost=self.layer3.negative_log_likelyhood(self.y)
		self.error=self.layer3.errors(self.y)

		self.debug=self.cost

		self.params=self.layer3.param+self.layer2.param+self.layer1.param+self.layer0.param
		self.grads=T.grad(self.cost,self.params)

		self.updates=[
			(param_i,param_i-self.learn_rate*grad_i*(0.8**(self.iter/3)))
			for param_i, grad_i in zip(self.params,self.grads)
		]
		print 'construction completed!'

	# def debug_func(self,data_x,data_y):
	# 	'''
	# 	>>>debugging function
	# 	'''
	# 	print 'begin'
	# 	assert data_x.shape[0]==data_y.shape[0]

	# 	data_batches=data_y.shape[0]/self.batch_size

	# 	dataX=theano.shared(data_x,borrow=True)
	# 	dataY=theano.shared(data_y,borrow=True)

	# 	index=T.iscalar('index')
	# 	debug_model=theano.function(
	# 		[index],
	# 		self.debug,
	# 		updates=self.updates,
	# 		givens={
	# 		self.x:dataX[index*self.batch_size:(index+1)*self.batch_size],
	# 		self.y:dataY[index*self.batch_size:(index+1)*self.batch_size]
	# 		}
	# 		)

	# 	print 'debug model established'

	# 	epoch=0
	# 	max_iter=1000000
	# 	done_looping=False
	# 	while (epoch<self.n_epochs) and (not done_looping):
	# 		epoch+=1
	# 		for batch_index in xrange(data_batches):
	# 			iter_num=(epoch-1)*data_batches+batch_index
	# 			if iter_num%100==0:
	# 				print 'training@iter=%d/%d' %(iter_num,min(max_iter,self.n_epochs*data_batches))

	# 			current_cost=debug_model(batch_index)
	# 			# print current_cost
	# 			if max_iter<=iter_num:
	# 				done_looping=True
	# 				break

	# 	print 'debug process complete!'

	# def train(self,train_set_x,train_set_y):
	# 	'''
	# 	>>>type train_set_x: T.dmatrix
	# 	>>>para train_set_x: data of instances of training set

	# 	>>>type train_set_y: T.ivector
	# 	>>>para train_set_y: labels of instances of training set
	# 	'''
	# 	assert train_set_x.shape[0]==train_set_y.shape[0]

	# 	train_set_batches=train_set_y.shape[0]/self.batch_size

	# 	trainX=theano.shared(train_set_x,borrow=True)
	# 	trainY=theano.shared(train_set_y,borrow=True)

	# 	index=T.iscalar('index')
	# 	train_model=theano.function(
	# 		[index],
	# 		self.cost,
	# 		updates=self.updates,
	# 		givens={
	# 		self.x:trainX[index*self.batch_size:(index+1)*self.batch_size],
	# 		self.y:trainY[index*self.batch_size:(index+1)*self.batch_size]
	# 		}
	# 		)

	# 	print 'training model established'

	# 	epoch=0
	# 	max_iter=1000000
	# 	done_looping=False
	# 	while (epoch<self.n_epochs) and (not done_looping):
	# 		epoch+=1
	# 		for batch_index in xrange(train_set_batches):
	# 			iter_num=(epoch-1)*train_set_batches+batch_index
	# 			if iter_num%100==0:
	# 				print 'training@iter=%d/%d' %(iter_num,min(max_iter,self.n_epochs*train_set_batches))

	# 			current_cost=train_model(batch_index)
	# 			if max_iter<=iter_num:
	# 				done_looping=True
	# 				break

	# 	print 'training process complete!'

	# def test(self,test_set_x,test_set_y):
	# 	'''		
	# 	>>>type test_set_x: T.dmatrix
	# 	>>>para test_set_x: data of instances of testing set

	# 	>>>type test_set_y: T.ivector
	# 	>>>para test_set_y: labels of instances of testing set
	# 	'''
	# 	assert test_set_x.shape[0]==test_set_y.shape[0]
	# 	testX=theano.shared(test_set_x,borrow=True)
	# 	testY=theano.shared(test_set_y,borrow=True)

	# 	test_set_batches=test_set_y.shape[0]/self.batch_size

	# 	index=T.iscalar('index')
	# 	test_model=theano.function(
	# 		[index],
	# 		self.error,
	# 		givens={
	# 		self.x: testX[index*self.batch_size:(index+1)*self.batch_size],
	# 		self.y: testY[index*self.batch_size:(index+1)*self.batch_size]
	# 		}
	# 		)

	# 	print 'test model established!'

	# 	test_errors=[
	# 	test_model(i)
	# 	for i in xrange(test_set_batches)
	# 	]
	# 	test_score=1.0-np.mean(test_errors)

	# 	print 'test accuracy: %f'%test_score

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
		epoch_num=T.iscalar('epoch_num')

		train_model=theano.function(
			[index, epoch_num],
			self.cost,
			updates=self.updates,
			givens={
			self.x:trainX[index*self.batch_size:(index+1)*self.batch_size],
			self.y:trainY[index*self.batch_size:(index+1)*self.batch_size],
			self.iter:epoch_num
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
			self.x: testX[index*self.batch_size:(index+1)*self.batch_size],
			self.y: testY[index*self.batch_size:(index+1)*self.batch_size]
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

		while (epoch<self.n_epochs) and (not done_looping):
			epoch+=1
			for batch_index in xrange(train_set_batches):
				iter_num=(epoch-1)*train_set_batches+batch_index
				#print self.layer0.w.get_value()[0,0,0,0]
				if iter_num%100==0:
					print 'training@iter=%d/%d'%(iter_num,train_set_batches*self.n_epochs)
				cost_now=train_model(batch_index,epoch)

				if (iter_num+1)%validate_fre==0:
					validation_losses=[
					validate_model(i)
					for i in xrange(validate_set_batches)
					]
					validation_loss_mean=np.mean(validation_losses)
					print 'epoch %i, batch_index%i/%i, validation accuracy %f %%'%(epoch,batch_index+1,train_set_batches,(1.0-validation_loss_mean)*100.)
					if validation_loss_mean<best_validation_loss:
						if validation_loss_mean<best_validation_loss*improvement_threshold:
							max_iter=max(max_iter,iter_num*max_iter_increase)

						best_validation_loss=validation_loss_mean
						best_iter=iter_num

						test_scores=[
						test_model(i)
						for i in xrange(test_set_batches)
						]
						test_score_mean=np.mean(test_scores)
						print 'epoch %i, batch_index %i/%i, test accuracy %f %%'%(epoch,batch_index+1,train_set_batches,test_score_mean*100.)

				if iter_num>=max_iter:
					done_looping=True
					break

		print 'best validate accuracy of %f %% at iteration %i, with test accuracy %f %%'%((1.0-best_validation_loss)*100.,best_iter+1,test_score_mean*100.)
