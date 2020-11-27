# ...

exit()

# This is needed
tf_all_embeddings = tf.constant(all_embeddings,dtype=tf.float32)
es = tf.shape(tf_all_embeddings)
tf_all_embeddings = tf.reshape(tf_all_embeddings, shape=(1,es[0],es[1]))

# Implementation with tensorflow
def neighbor_accuracy_impl_tf(y_true, y_pred):
    s = tf.shape(y_pred)
    y_pred = tf.reshape(y_pred, shape=(s[0],1,s[1]))
    
    # diff has shape [batch_size, all_nodes, embedding_dim]
    diff = tf.subtract(tf_all_embeddings, y_pred)
    #print("diff.shape:",diff.shape)
    
    # take norm with respect to embedding_dim
    diff_norm = tf.norm(diff, axis=2)
    #print("diff_norm.shape:",diff_norm.shape)
    
    # find argmin for each of elements in batch
    idxs = tf.math.argmin(diff_norm, axis=1)
    #print("idxs.shape:",idxs.shape)
    
    y_pred_embs = tf.gather(tf.squeeze(all_embeddings),idxs)
    #print("y_pred_embs.shape:",tf.shape(y_pred_embs))
    
    results = tf.cast(tf.math.equal(y_pred_embs,y_true),tf.float32)
    results = tf.norm(results,axis=1)
    #print(results)
    
    return tf.math.reduce_sum(results) / tf.cast(tf.shape(results)[0],tf.float32)
