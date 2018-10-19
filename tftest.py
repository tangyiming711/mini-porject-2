#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage import io,transform
import tensorflow as tf
import numpy as np

path1 = "C:/111/zzz.jpg"  #image address
path2 = "C:/111/zz.jpg" 
 
flower_dict = {1:'roses',2:'sunflowers'}    #classification 
w=100 
h=100 
c=3 
def read_one_image(path): 
    img = io.imread(path) 
    img = transform.resize(img,(w,h)) 
    return np.asarray(img) 

with tf.Session() as sess: 
    data = [] 
    data1 = read_one_image(path1) 
    data2 = read_one_image(path2) 
    data.append(data1) 
    data.append(data2) 

    saver = tf.train.import_meta_graph('C:/1/flower_photos/model2.ckpt.meta')  #model address 
    saver.restore(sess,tf.train.latest_checkpoint('C:/1/flower_photos/'))      #data address

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}
    
    logits = graph.get_tensor_by_name("logits_eval:0")
    
    classification_result = sess.run(logits,feed_dict)
    
    #Print predicted matrix
    print(classification_result)
    
    #Print an index of the maximum value of each row of the prediction matrix
    print(tf.argmax(classification_result,1).eval())
    
    #classfication
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("flower",i+1,"prediction:"+flower_dict[output[i]])
    


# In[ ]:




