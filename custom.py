#class EmbeddingModel:
    #def __init__(self,num_categories,embedding_dim,dense_layers):
        ## embeddes the ids in a higher-dimensional space
        #self.embedding_network = make_embedding_network(num_categories,embedding_dim)
        
        ## two surface embeddings: result [0,1] How likely these surfaces are connected?
        #self.network = make_keras_mlp(2*embedding_dim, dense_layers, 1, output_activation=tf.nn.sigmoid)
        
        ## store variables
        #self.variables = self.embedding_network.trainable_variables + self.network.trainable_variables
        
        
    #def forward(self,start_surface_ids,end_surface_ids):
        #assert start_surface_ids.shape == end_surface_ids.shape
        
        ## create embeddings
        #start_embeddings = self.embedding_network(start_surface_ids)
        #end_embeddings = self.embedding_network(end_surface_ids)
        
        ## combine input data
        #input_data = tf.concat([start_embeddings,end_embeddings],1)
        
        #return self.network(input_data)
    
    
    #def train(self,x_train,y_train,epochs,batchsize=32,validation_split=0.33,learning_rate=0.01):
        #assert len(x_train) == len(y_train)
        #loss_data = []
        
        #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        #x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=validation_split)
        #train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(train_x)).batch(batchsize)
        #validation_data = tf.data.Dataset.from_tensor_slices((x_val,y_val)).shuffle(len(x_val))
        
        #mse = tf.keras.losses.MeanSquaredError()
        
        #for epoch in range(epochs):
            #epoch_losses = []
            
            #for x,y_true in train_data:
                #x = tf.transpose(x)
                
                #with tf.GradientTape() as tape:
                    #y_pred = self.forward(x[0],x[1])
                    #loss = mse(y_true,y_pred)
                
                #grad = tape.gradient(loss, self.variables)
                #optimizer.apply_gradients(zip(grad, self.variables))
                
                #epoch_losses.append(loss)
                    
            #for x,y in validation_data:
                #y_pred = self.forward(x[0],x[1])
                #epoch_losses.append( mse(y_true,y_pred) )
            
            #if epoch % 1 == 0:
                #print("train loss:",epoch_losses[0],"validation loss:",epoch_losses[1],flush=True)
            
            #loss_data.append(epoch_losses)
            
        #return loss_data
