
# LensFlare

LensFlare is an example package I created to help myself and others better understand neural networks. A lot of the code is based off work that I did in the [Coursera deeplearning.ai course](https://www.coursera.org/specializations/deep-learning)

An example work flow is shown below:


```python
import tensorflow as tf
from lensflare.classification import TfNNClassifier
from lensflare.util import load_moons_dataset
```


```python
X_train, y_train = load_moons_dataset()
```


![png](readme_files/plot_data.png)



```python
tf.reset_default_graph()

# layer_dims contains neural network structure parameters
layers_dims=[X_train.shape[0], 200, 80, 10, 1]
clf = TfNNClassifier(layers_dims=layers_dims,
                  optimizer="adam",
                  lambd=.05,
                  keep_prob=0.7,
                  num_epochs=5000)
clf.fit(X_train, y_train, seed=3)
y_pred_train = clf.transform(X_train, y_train)
```

    Cost after epoch 0: 1.036825
    Cost after epoch 1000: 0.108737
    Cost after epoch 2000: 0.104837
    Cost after epoch 3000: 0.106805
    Cost after epoch 4000: 0.105311
    INFO:tensorflow:Restoring parameters from results/model
    Training Accuracy: 0.983333333333



```python
from lensflare.funcs.tf_funcs import plot_decision_boundary, predict_dec
# Plot decision boundary

predictions, X, dropout_var, sess = predict_dec()
model = lambda X_train: sess.run([predictions], feed_dict={X:X_train, dropout_var: 1.0});

plot_decision_boundary(model, X_train, y_train)
sess.close()
```

    INFO:tensorflow:Restoring parameters from results/model



![png](readme_files/decision_boundary.png)
