import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

from  sklearn.datasets.samples_generator import make_blobs 
from  sklearn.datasets.samples_generator import make_circles 

DATA_TYPE="blobs"
#DATA_TYPE="cirlce"
#number of clusters,if we choose circles,only 2 will enough
if DATA_TYPE=="cicle":
    K=2
else:
    K=4
#Maximum number of iteration,if the conditions are not met
MAX_ITERS=1000
#Number of samples
N=200
start=time.time()
centers=[(-2,-2),(-2,1.5),(1.5,-2),(2,1.5)]
if DATA_TYPE=="circle":
    data,features=make_circles(n_samples=N,shuffle=True,
                               noise=0.01,factor=0.4)
else:
    data,features=make_blobs(n_samples=N,centers=centers,n_features=2,
                            cluster_std=0.8,shuffle=False,random_state=42)
fig=plt.figure()
ax1=fig.add_subplot(121)
ax1.scatter(np.asarray(centers).transpose()[0],np.asarray(centers).transpose()[1],marker="o",s=200,color="r")
#plt.show()
points=tf.Variable(data)
cluster_assignment=tf.Variable(tf.zeros([N],dtype=tf.int64))
centroids=tf.Variable(tf.slice(points.initialized_value(),[0,0],[K,2]))

rep_points=tf.reshape(tf.tile(points,[1,K]),[N,K,2])
rep_centroids=tf.reshape(tf.tile(centroids,[N,1]),[N,K,2])
sum_squares=tf.reduce_sum(tf.square(rep_points-rep_centroids),
                          reduction_indices=2)
best_centroids=tf.argmin(sum_squares,1)
did_assignment_changed=tf.reduce_any(tf.not_equal(
    cluster_assignment,best_centroids))

def bucket_mean(data,bucket_ids,num_buckets):
    total=tf.unsorted_segment_mean(data,bucket_ids,num_buckets)
    count=tf.unsorted_segment_mean(tf.ones_like(data),
                                   bucket_ids,num_buckets)
    return total/count
means=bucket_mean(points,best_centroids,K)


with tf.control_dependencies([did_assignment_changed]):
    do_updates=tf.group(centroids.assign(means),
            cluster_assignment.assign(best_centroids))
sess=tf.Session()
sess.run(tf.global_variables_initializer())
#print sess.run(means)
changed=True
iter=0
while changed and iter<MAX_ITERS:
    #fig,ax=plt.subplots()
    iter+=1
    [changed,_]=sess.run([did_assignment_changed,do_updates])
    [centers,assignments]=sess.run([centroids,cluster_assignment])


ax2=fig.add_subplot(122)
ax2.scatter(sess.run(points).transpose()[0],
            sess.run(points).transpose()[1],
            marker="o",c=assignments,s=220,cmap=plt.get_cmap('coolwarm'))
ax2.scatter(centers[:,0],centers[:,1],
           marker="^",s=550)
ax2.set_title("Iteration "+str(iter))
plt.show()
end=time.time()
print "Found in %.2f seconds" % (end-start),iter," iterations"
print "centroids:"
print centers
print "Cluster assignments: ",assignments
