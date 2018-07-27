import tensorflow as tf
import os
from tensorflow.python.client import timeline


w = tf.Variable(2.0,name='w')
b = tf.Variable(3.0,name='bias')
#if x,y is variable , they will also be updated by optimizer or declare trainable=false
#x = tf.Variable(5.0,name='x')
x = tf.Variable(5.0,trainable=False,name='x')
#y = tf.Variable(8.0,name='y')
y = tf.Variable(8.0,trainable=False,name='y')
#x = tf.placeholder(tf.float32,name='x')
#y = tf.placeholder(tf.float32,name='y')

#mul_result =  tf.multiply(w,x)
#pred = tf.add(mul_result,b)
#no need to use function
#with tf.device('a'):
pred = w * x + b
loss = tf.square(y-pred,name='loss')
gradients = tf.gradients(pred,x)
gradients2 = tf.gradients(pred,b)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#summary related code
tf.summary.scalar("loss",loss)
tf.summary.histogram("histogram",loss)
summary_op = tf.summary.merge_all()

#not use
#gradient = tf.gradients(pred,w)

#create a saver object
saver = tf.train.Saver()
ckpt_dir = 'ckp_dir/model'
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

run_metadata = tf.RunMetadata()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #check if there is checkpoint
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir))
    #check if there is a valid checkpoint path
    if ckpt  and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)

    for i in range(30):
        #print(sess.run([optimizer,loss,w,x,y,pred],feed_dict={x:2,y:3}))
        o_result,loss_result,w_result,x_result,y_result,pred_result,summary = sess.run([optimizer,loss,w,x,y,pred,summary_op],options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
        print(sess.run(gradients))
        print(sess.run(gradients2))
        print(o_result,loss_result,w_result,x_result,y_result,pred_result)
        writer.add_summary(summary, global_step=i)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        if (i+1) % 10 == 0:
            saver.save(sess,ckpt_dir,global_step=i)
        #print(w.eval())
trace_file = open('timeline.ctf.json','w')
trace_file.write(trace.generate_chrome_trace_format())
writer.close()
