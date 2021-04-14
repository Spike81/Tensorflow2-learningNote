# Tensorflow
## 0. 前期准备
```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)
```
## 1. 数据类型
### 1. 创建 int, float, double, boolean, String 类型

```python
tf.constant(1)  # 创建值为 1 的 int32 型数据
tf.constant(1.)  # # 创建值为 1.0 的 float32 型数据
tf.constant(2.2, dtype = tf.double)
tf.constant([True, False])  # boolean
tf.constant("hello, world")  # String
```

### 2. Tensor Property
```python
with tf.device("cpu"):
    a = tf.constant([1])  # 使用 cpu 创建 tensor
print(a, a.device)  # tf.Tensor([1], shape=(1,), dtype=int32) /job:localhost/replica:0/task:0/device:CPU:0

with tf.device("gpu"):
    b = tf.range(4)  # 使用 cpu 创建 tensor
print(b, b.device)  # tf.Tensor([0 1 2 3], shape=(4,), dtype=int32) /job:localhost/replica:0/task:0/device:GPU:0
```
```python
print(b)  # tf.Tensor([0 1 2 3], shape=(4,), dtype=int32)
print(b.numpy())  # [0 1 2 3], tensor 转 numpy
print(b.ndim)  # 1, b 的维度
print(tf.rank(b))  # tf.Tensor(1, shape=(), dtype=int32), 也是返回 b 的维度，但类型是 tensor
```
### 3. Check Tensor Type
```python
a = tf.constant(2.)
b = tf.constant([True, False])
c = tf.constant("hello, world")

print(tf.is_tensor(a))  # True
print(a.dtype, b.dtype, c.dtype)  # <dtype: 'float32'> <dtype: 'bool'> <dtype: 'string'>
print(c.dtype == tf.string)  # True
```
### 4. Convert
```python
a = np.arange(5)
print(a,)  # [0 1 2 3 4]
 # 将其他数据转为 tensor
aa = tf.convert_to_tensor(a, dtype=tf.int32)
print(aa)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)

# 在 tensor 的不同类型之间转换
b = tf.cast(aa, dtype=tf.float32)
print(b)  # tf.Tensor([0. 1. 2. 3. 4.], shape=(5,), dtype=float32)
bb = tf.cast(b, dtype=tf.double)
print(bb)  # tf.Tensor([0. 1. 2. 3. 4.], shape=(5,), dtype=float64)

b = tf.constant([0, 1])
bb = tf.cast(b, dtype=tf.bool)
print(bb)  # tf.Tensor([False  True], shape=(2,), dtype=bool)
bbb = tf.cast(bb, dtype=tf.int32)
print(bbb)  # tf.Tensor([0 1], shape=(2,), dtype=int32)
```
### 5. tf.Variable
```python
a = tf.range(5)
b = tf.Variable(a, name="input_data")
print(b.name, b.trainable, tf.is_tensor(b))  # input_data:0 True True
```
## 2. 创建 tensor
### 1. 通过转换 numpy，list 创建
```python
a = tf.convert_to_tensor(np.ones([2, 3]))
print(a)  # tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]], shape=(2, 3), dtype=float64)

b = tf.convert_to_tensor(np.zeros([2, 3]))
print(b)  # tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float64)

c = tf.convert_to_tensor([1, 2])
print(c)  # tf.Tensor([1 2], shape=(2,), dtype=int32)

d = tf.convert_to_tensor([[1], [2.]])
print(d)  # tf.Tensor(
[[1.]
 [2.]], shape=(2, 1), dtype=float32)
```
### 2. 直接创建
#### 1. tf.zeros
```python
a = tf.zeros([])
print(a, a.shape)  # tf.Tensor(0.0, shape=(), dtype=float32) ()

b = tf.zeros([1])
print(b, b.shape)  # tf.Tensor([0.], shape=(1,), dtype=float32) (1,)

c = tf.zeros([2, 2, 3])
print(c, c.shape)  # tf.Tensor(
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]], shape=(2, 2, 3), dtype=float32) (2, 2, 3)
```

#### 2. tf.zeros_like
```python
a = tf.zeros([2, 3, 3])

b = tf.zeros_like(a)
print(b, b.shape)  # tf.Tensor(
[[[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]], shape=(2, 3, 3), dtype=float32) (2, 3, 3)

c = tf.zeros(a.shape)
print(c, c.shape)  # tf.Tensor(
[[[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 0. 0.]]], shape=(2, 3, 3), dtype=float32) (2, 3, 3), 和zeros_like 效果一样
```
#### 3. tf.ones and tf.ones_like
```python
a = tf.ones([])  # tf.Tensor(1.0, shape=(), dtype=float32)
b = tf.ones(1)  # tf.Tensor([1.], shape=(1,), dtype=float32)
c = tf.ones([2, 3])  # tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]], shape=(2, 3), dtype=float32)
 
d = tf.ones_like(c)  # tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]], shape=(2, 3), dtype=float32)
e = tf.ones(c.shape)  # tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]], shape=(2, 3), dtype=float32)
```
#### 4. tf.fill
```python
a = tf.fill([2,2], 0)  # tf.Tensor(
[[0 0]
 [0 0]], shape=(2, 2), dtype=int32)
b = tf.fill([2,1], 9.1)  # tf.Tensor(
[[9.1]
 [9.1]], shape=(2, 1), dtype=float32)
```
#### 5. tf.random
##### 1. 正态分布
```python
a = tf.random.normal([2,2], mean=1, stddev=1)
# tf.Tensor(
[[1.1882732  0.00315791]
 [2.6843395  1.2440206 ]], shape=(2, 2), dtype=float32)
# mean: 均值; stddev: 标准差，表示组内个体间的离散程度

b = tf.random.truncated_normal([2,2], mean=1, stddev=1)
# tf.Tensor(
[[-0.3323753  -0.12528265]
 [-0.4473424   1.2448564 ]], shape=(2, 2), dtype=float32)
# 和上面不同的是，这个将一个函数截断，只取部分。性能较 tf.random.normal 更好
```
##### 2. 均匀分布
```python
a = tf.random.uniform([2,2], minval=0, maxval=1)
print(a)  # tf.Tensor(
[[0.4400009  0.17153084]
 [0.15190983 0.3065387 ]], shape=(2, 2), dtype=float32)
```
##### 3. Random Permutation (随机打散)
```python
idx = tf.range(10)
print(idx)  # tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)
idx = tf.random.shuffle(idx)
print(idx)  # tf.Tensor([2 4 0 5 6 7 3 9 1 8], shape=(10,), dtype=int32)
```
##### 4. Loss

```python
out = tf.random.uniform([4, 10])
y = tf.range(4)
y = tf.one_hot(y, depth=10)

loss = tf.keras.losses.mse(y, out)  # mse: ∑(y - out)^2
loss = tf.reduce_mean(loss)
```
##### 5. kernel and bias
```python
net = layers.Dense(10)  # construct network
net.build((4, 8))  # set the input shape
print(net.kernel)  # kernel is like w in y = w@x + b, its shape is [8,10] here
print(net.bias)  # bias is like b in y = w@x + b, its shape is [10] here

a = tf.random.uniform([4, 8])
out = net(a)
print(out)  # shape of out is [4,10]
```
## 索引和切片
### Basic indexing
```python
a = tf.ones([1,5,5,3])
print(a[0].shape, a[0][0].shape, a[0][0][0].shape)  # (5, 5, 3) (5, 3) (3,)
print(a[0][0][0][1])  # tf.Tensor(1.0, shape=(), dtype=float32)
```
### Numpy-style indexing
```python
b = tf.ones([1,5,5,3])
print(b[0].shape, b[0,0].shape, b[0,0,0].shape)  # (5, 5, 3) (5, 3) (3,)
print(b[0,0,0,1])  # tf.Tensor(1.0, shape=(), dtype=float32)
```
### 切片
```python
a = tf.range(10)
print(a)  # tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)

print(a[:2])  # tf.Tensor([0 1], shape=(2,), dtype=int32)
print(a[-1:-5:-1])  # tf.Tensor([9 8 7 6], shape=(4,), dtype=int32)
```
```python
a = tf.ones([4,28,28,3])
print(a[0].shape)  # (28, 28, 3)
print(a[0,:,:,:].shape)  # (28, 28, 3)
print(a[0,1,0:5,:].shape)  # (5, 3)
print(a[:,:,:,2].shape)  # (4, 28, 28)
print(a[:,::2,::10,2].shape)  # (4, 14, 3)
```
### ... 在切片中的使用
```python
b = tf.ones([2,4,28,28,3])
print(b[1,:,:,:,0].shape)  # (4, 28, 28)
print(b[1,...,0].shape)  # (4, 28, 28)
print(b[...,0,0].shape)  # (2, 4, 28)
```
### Selective Indexing
#### tf.gather
```python
a = tf.random.uniform([4,35,8])  # [class,student,subject]

b = tf.gather(a, axis=0, indices=[2,3])  # means select the number 2 and 3 of "class"
print(b.shape)  # (2, 35, 8)
c = tf.gather(a, axis=0, indices=[2,1,3,0])  # means re-arrange the "class" as 2,1,3,0
print(c.shape)  # (4, 35, 8)
d = tf.gather(a, axis=1, indices=[1,3,5,7,9])  # means select 5 students
print(d.shape)  # (4, 5, 8)
e = tf.gather(a, axis=2, indices=[2,8])  # means select 2 subjects
print(e.shape)  # (4, 35, 2)
```
#### tf.gather_nd
```python
a = tf.random.uniform([4,35,8])  # [class,student,subject]
# 如果想看 2 个 class 中的 5 个同学的所有 subject
aa = tf.gather(a, axis=0, indices=[1,4])
aaa = tf.gather(aa, axis=1, indices=[1,2,5,7,11])
print(aaa.shape)  # (2, 5, 8)
```
```python
# tf.gather_nd 使用示范
a = tf.random.normal([2,2,3])
print(a)  # tf.Tensor(
[[[-1.0109025  -0.06776652  0.5304096 ]
  [ 0.45643803 -0.22623973  1.9866889 ]]

 [[-1.6605937  -0.931652    1.4222666 ]
  [-0.3711534  -0.00418805  0.426787  ]]], shape=(2, 2, 3), dtype=float32)

b = tf.gather_nd(a, [0])
print(b)  # tf.Tensor(
[[-1.0109025  -0.06776652  0.5304096 ]
 [ 0.45643803 -0.22623973  1.9866889 ]], shape=(2, 3), dtype=float32), 相当于 a[0]

c = tf.gather_nd(a, [0,1])
print(c)  # tf.Tensor([ 0.45643803 -0.22623973  1.9866889 ], shape=(3,), dtype=float32)，相当于 a[0,1]

d = tf.gather_nd(a, [0,1,2])
print(d)  # tf.Tensor(1.9866889, shape=(), dtype=float32)

e = tf.gather_nd(a, [[0,1,2]])
print(e)  # tf.Tensor([1.9866889], shape=(1,), dtype=float32)，相当于 [a[0,1,2]]
```
```python
a = tf.random.normal([4,35,8])
# 如果想看 2 个 class 中的 5 个同学的所有 subject
b = tf.gather_nd(a, [[1,2],[1,5], [0,0],[0,7],[0,11]])  # c1s2,c1s5,c0s0,c0s7,c0s11
print(b.shape)  # (5, 8)
```
#### tf.boolean_mask
```python
# tf.boolean_mask 只选取为 True 的元素
a = tf.random.normal([4,28,28,3])
b = tf.boolean_mask(a, mask=[True,True,False,False])  # axis 默认为 0, mask 里的数量要和 axis 的数量对应
print(b.shape)  # (2, 28, 28, 3)
c = tf.boolean_mask(a, mask=[True,True,False], axis=3)
print(c.shape)  # (4, 28, 28, 2)

aa = tf.ones([2,3,4])
bb = tf.boolean_mask(aa, mask=[True,False])  # 此时的 axis 为 0
print(bb.shape)  # (1, 3, 4)
cc = tf.boolean_mask(aa, mask=[[True,False,False],[False,True,True]])  # 因为外面还有一个括号，所以 axis 为 1
print(cc.shape)  # (3, 4)
```

## 维度变换
### Reshape
```python
a = tf.random.normal([4,28,28,3])
b = tf.reshape(a, [4,-1,3])
print(b.shape)  # (4, 784, 3)
```
### tf.transpose
```python
a = tf.random.normal([4,3,2,1])
b = tf.transpose(a)
print(b.shape)  # (1, 2, 3, 4)
c = tf.transpose(a, perm=[1,0,3,2])
print(c.shape)  # (3, 4, 1, 2)
```
### Squeeze VS Expand_dims
#### Expand dim
```python
a = tf.random.normal([4,35,8])
b = tf.expand_dims(a, axis=0)
print(b.shape)  # (1, 4, 35, 8)
c = tf.expand_dims(a, axis=2)
print(c.shape)  # (4, 35, 1, 8)
d = tf.expand_dims(a, axis=3)
print(d.shape)  # (4, 35, 8, 1)

# axis 给正数，在该数前面增加一维；axis 给负数，在该数后面增加一维
e = tf.expand_dims(a, axis=-1)
print(e.shape)  # (4, 35, 8, 1)
f = tf.expand_dims(a, axis=-3)
print(f.shape)  # (4, 1, 35, 8)
```

#### Squeeze dim
```python
# 把维度为 1 的去除
a = tf.squeeze(tf.zeros([1,2,1,1,3]))
print(a.shape)  # (2, 3)

b = tf.squeeze(tf.zeros([1,2,1,1,3]), axis=2)
print(b.shape)  # (1, 2, 1, 3)
c = tf.squeeze(b, axis=-2)
print(c.shape)  # (1, 2, 3)
```

### Broadcasting
例如 [b, 10] + [10] 这样的运算，因为 shape 不相等，不能直接相加，因此实际上是先进行了 broadcasting，把 [10] 变成了 [b, 10] 。
```python
a = tf.random.normal([4,32,32,3])
b = a + tf.random.normal([4,32,32,1])
print(b.shape)  # (4, 32, 32, 3)
c = a + tf.random.normal([3])
print(c.shape)  # (4, 32, 32, 3)
d = a + tf.random.normal([4,1,1,1])
print(d.shape)  # (4, 32, 32, 3)
e = a + tf.random.normal([1,4,1,1])
print(e.shape)  # error, 因为只有相同位置为 1 或相同时，才能 broadcasting

f = tf.random.normal([4,32,32,3])
g = tf.random.normal([4,1,1,1])
h = tf.broadcast_to(g, f.shape)
print(h.shape)  # (4, 32, 32, 3)
```

### Tile
Broadcast 虚拟增维，不会占用内存；而 tile 是真实的增加维度，会占用内存。
```python
a = tf.ones([3,4])
b = tf.broadcast_to(a, [2,3,4])
print(b.shape)  # (2, 3, 4)

c = tf.expand_dims(a, axis=0)
print(c.shape)  # (1, 3, 4)
c = tf.tile(c, [2,1,1])  # [2,1,1] 代表将 c 的第一维度复制到两倍，二三维度不变
print(c.shape)  # (2, 3, 4)
```

## 数学运算
加减乘除：+, -, *, /
n 次方 和 开方：**, pow, square, sqrt
整除 和 取余：//, %
对数运算：exp, log
矩阵运算：@, matmul

### 加减乘除，整除，取余
```python
a = tf.ones([2,2])
b = tf.fill([2,2],2.)

print(a + b)  # tf.Tensor(
[[3. 3.]
 [3. 3.]], shape=(2, 2), dtype=float32)
 
print(a - b)  # tf.Tensor(
[[-1. -1.]
 [-1. -1.]], shape=(2, 2), dtype=float32)
 
print(a * b)  # tf.Tensor(
[[2. 2.]
 [2. 2.]], shape=(2, 2), dtype=float32)
 
print(a / b)  # tf.Tensor(
[[0.5 0.5]
 [0.5 0.5]], shape=(2, 2), dtype=float32)
 
print(b // a)  # tf.Tensor(
[[2. 2.]
 [2. 2.]], shape=(2, 2), dtype=float32)

print(b % a)  # tf.Tensor(
[[0. 0.]
 [0. 0.]], shape=(2, 2), dtype=float32)
```

### 对数运算
```python
# 以 e 为底
a = tf.ones([2,2])

b = tf.math.log(a)  # log1 = 0
print(b)  # tf.Tensor(
[[0. 0.]
 [0. 0.]], shape=(2, 2), dtype=float32)

c = tf.exp(a)  # lg1 = 2,718
print(c)  # tf.Tensor(
[[2.7182817 2.7182817]
 [2.7182817 2.7182817]], shape=(2, 2), dtype=float32)
```

要表示例如 log~2~ 和 log~10~ 等，需要用到：**log~a~b / log~a~c = log~c~b**
```python
a = tf.math.log(8.) / tf.math.log(2.)
print(a)  # tf.Tensor(3.0, shape=(), dtype=float32)
b = tf.math.log(100.) / tf.math.log(10.)
print(b)  # tf.Tensor(2.0, shape=(), dtype=float32)
```

### n 次方 和 开方
```python
a = tf.fill([2,2], 2.)
print(tf.pow(a,3))  # tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)

print(a ** 3)  # tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)

print(tf.sqrt(a))  # tf.Tensor(
[[1.4142135 1.4142135]
 [1.4142135 1.4142135]], shape=(2, 2), dtype=float32)
```

### 矩阵运算
```python
a = tf.fill([2,2], 1.)
b = tf.fill([2,2], 2.)

print(a @ b)  # tf.Tensor(
[[4. 4.]
 [4. 4.]], shape=(2, 2), dtype=float32)

print(tf.matmul(a, b))  # tf.Tensor(
[[4. 4.]
 [4. 4.]], shape=(2, 2), dtype=float32)

# 下面这种情况代表 [2,3] @ [3,5]，进行 4 次
c = tf.fill([4,2,3], 1.)
d = tf.fill([4,3,5], 2.)

print(c @ d)  # (4, 2, 5)
print(tf.matmul(c, d))  # (4, 2, 5)
```

## 张量的合并与拼接
### tf.concat
tf.concat 可以将 tensor 进行合并
```python
a = tf.ones([4,35,8])
b = tf.ones([2,35,8])
c = tf.concat([a,b], axis=0)
print(c.shape)  # (6, 35, 8)

a = tf.ones([4,20,8])
b = tf.ones([4,15,8])
c = tf.concat([a,b], axis=1)
print(c.shape)  # (4, 35, 8)
```
### tf.stack
tf.stack 会合并 tensor，并增加一个维度
```python
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.stack([a,b], axis=0)
print(c.shape)  # (2, 4, 35, 8)

d = tf.stack([a,b], axis=2)
print(d.shape)  # (4, 35, 2, 8)
```
### tf.unstack
tf.unstack 是 tf.stack 的逆操作，会将 tensor 拆分，并减去一个维度
```python
a = tf.ones([2,4,35,8])
aa, bb = tf.unstack(a, axis=0)
print(aa.shape, bb.shape)  # (4, 35, 8) (4, 35, 8)

c = tf.unstack(a, axis=3)  # 会将 a 拆成 8 个 tensor
print(len(c), c[0].shape, c[7].shape)  # 8 (2, 4, 35) (2, 4, 35)
```
### tf.split
tf.split 比 tf.unstack 更加灵活，可以指定拆分的数量
```python
a = tf.ones([2,4,35,8])
b = tf.split(a, axis=3, num_or_size_splits=2)
print(len(b), b[1].shape)  # 2 (2, 4, 35, 4)

c = tf.split(a, axis=3, num_or_size_splits=[2,2,4])
print(len(c), c[2].shape)  # 3 (2, 4, 35, 4)
```

## 数据统计
### tf.norm (范数)
1-范数：所有元素 的 绝对值 的 和
2-范数：将所有元素 的 平方和 开方
```python
a = tf.ones([2,2])

b = tf.norm(a)  # 默认为 2-范数
print(b)  # tf.Tensor(2.0, shape=(), dtype=float32)

c = tf.norm(a, ord=2)  # ord 是 范数的类型
print(c)  # tf.Tensor(2.0, shape=(), dtype=float32)

d = tf.norm(a, ord=2, axis=1)  # axis 指定对哪个轴计算
print(d)  # tf.Tensor([1.4142135 1.4142135], shape=(2,), dtype=float32)

e = tf.norm(a, ord=1)
print(e)  # tf.Tensor(4.0, shape=(), dtype=float32)

f = tf.norm(a, ord=1, axis=0)
print(f)  # tf.Tensor([2. 2.], shape=(2,), dtype=float32)
```

### reduce_min/max/mean
这些操作会降维，reduce 正是此意
```python
a = tf.random.truncated_normal([4,10])
print(tf.reduce_max(a), tf.reduce_min(a), tf.reduce_mean(a))  # tf.Tensor(1.742887, shape=(), dtype=float32) tf.Tensor(-1.7156912, shape=(), dtype=float32) tf.Tensor(0.05339384, shape=(), dtype=float32)

print(tf.reduce_max(a, axis=1))  # tf.Tensor([1.2385428 1.742887  1.1014041 0.7983043], shape=(4,), dtype=float32)
```

### tf.argmax/tf.argmin
返回最大值/最小值的 index
```python
a = tf.random.normal([4, 10])
print(tf.argmax(a).shape)  # (10,)
print(tf.argmin(a).shape)  # (10,)
```

### tf.equal
```python
a = tf.constant([1,2,3,4,5])
b = tf.range(5)
print(tf.equal(a,b))  # tf.Tensor([False False False False False], shape=(5,), dtype=bool)

res = tf.reduce_sum(tf.cast(tf.equal(a,b), tf.int32))
print(res)  # tf.Tensor(0, shape=(), dtype=int32)
```

### Accuracy
计算 accuracy 的方法
```python
a = tf.constant([[0.1, 0.2, 0.7],
                 [0.9, 0.05, 0.05]])
pred = tf.cast(tf.argmax(a, axis=1), dtype=tf.int32)
print(pred)  # tf.Tensor([2 0], shape=(2,), dtype=int32)

res = tf.constant([2, 1])
print(tf.equal(pred,res))  # tf.Tensor([ True False], shape=(2,), dtype=bool)

acc = tf.reduce_sum(tf.cast(tf.equal(pred,res), dtype=tf.int32)) / len(pred)
print(acc)  # tf.Tensor(0.5, shape=(), dtype=float64)
```

### tf.unique
```python
a = tf.constant([1,2,3,2,1,2,4])
print(tf.unique(a))  # Unique(y=<tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 2, 3, 4])>, idx=<tf.Tensor: shape=(7,), dtype=int32, numpy=array([0, 1, 2, 1, 0, 1, 3])>) 前一个 array 是去除相同元素后的结果，第二个 array 是第一个 array 的 index
```

## 张量排序
### sort, argsort
sort 会返回排列好的 tensor，而 argsort 会返回排列好的 index
```python
a = tf.random.shuffle(tf.range(5))
print(a)  # tf.Tensor([2 4 3 0 1], shape=(5,), dtype=int32)

print(tf.sort(a, direction="DESCENDING"))  # tf.Tensor([4 3 2 1 0], shape=(5,), dtype=int32)

print(tf.argsort(a, direction="DESCENDING"))  # tf.Tensor([1 2 0 4 3], shape=(5,), dtype=int32)

print(tf.gather(a, tf.argsort(a, direction="DESCENDING")))  # tf.Tensor([4 3 2 1 0], shape=(5,), dtype=int32)
```
### top_k
top_k 可以返回最大个前 k 个元素的 index 和 value
```python
a = tf.constant([[4,6,8],
                 [9,4,7],
                 [4,5,1]])

res = tf.math.top_k(a, 2)
print(res.indices)  # tf.Tensor(
[[2 1]
 [0 2]
 [1 0]], shape=(3, 2), dtype=int32)
print(res.values)  # tf.Tensor(
[[8 6]
 [9 7]
 [5 4]], shape=(3, 2), dtype=int32)
```

### softmax
softmax 可以使 tensor 内同一维度的数总和为 1
```python
a = tf.random.normal([2,2])
print(a)  # tf.Tensor(
[[ 0.8081271   0.08110407]
 [-1.4335581  -0.365184  ]], shape=(2, 2), dtype=float32)

a = tf.math.softmax(a, axis=1)
print(a)  # tf.Tensor(
[[0.67415166 0.32584834]
 [0.2557124  0.7442876 ]], shape=(2, 2), dtype=float32)
```

## 填充 与 复制
### pad
```python
a = tf.reshape(tf.range(9), [3,3])
print(a)  # tf.Tensor(
[[0 1 2]
 [3 4 5]
 [6 7 8]], shape=(3, 3), dtype=int32)

b = tf.pad(a, [[1,0],[0,1]])  # [[1,0],[0,1]] 代表 a 的行头 pad 1 行，行尾 pad 0 行；列左 pad 0 列，列右 pad 1 列
print(b)  # tf.Tensor(
[[0 0 0 0]
 [0 1 2 0]
 [3 4 5 0]
 [6 7 8 0]], shape=(4, 4), dtype=int32)

print(tf.pad(a, [[2,1],[1,2]]))  # tf.Tensor(
[[0 0 0 0 0 0]
 [0 0 0 0 0 0]
 [0 0 1 2 0 0]
 [0 3 4 5 0 0]
 [0 6 7 8 0 0]
 [0 0 0 0 0 0]], shape=(6, 6), dtype=int32)
 
# image padding
a = tf.random.normal([4,28,28,3])
b = tf.pad(a, [[0,0], [2,2], [2,2], [0,0]])
print(b.shape)  # (4, 32, 32, 3)
```
### tile
```python
a = tf.reshape(tf.range(4), [2,2])
print(tf.tile(a, [1,2]))  # tf.Tensor(
[[0 1 0 1]
 [2 3 2 3]], shape=(2, 4), dtype=int32)

print(tf.tile(a, [2,1]))  # tf.Tensor(
[[0 1]
 [2 3]
 [0 1]
 [2 3]], shape=(4, 2), dtype=int32)

print(tf.tile(a, [2,2]))  # tf.Tensor(
[[0 1 0 1]
 [2 3 2 3]
 [0 1 0 1]
 [2 3 2 3]], shape=(4, 4), dtype=int32)
```

