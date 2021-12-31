import time
import jittor as jt
from jittor.models import resnet50
jt.flags.use_cuda = jt.has_cuda

warmup = 10
rerun = 100
batch_size = 8
data = jt.random((batch_size, 3, 224, 224))

# 此段代码对jittor进行热身，确保时间测试准确
jt.sync_all(True)
for i in range(warmup):
    pred = upfirdn2d(data)
    # sync是把计算图发送到计算设备上
    pred.sync()
# sync_all(true)是把计算图发射到计算设备上，并且同步。
# 只有运行了jt.sync_all(True)才会真正地运行，时间才是有效的，因此执行forward前后都要执行这句话
jt.sync_all(True)

# 开始测试运行时间
start = time.time()
for i in range(rerun):
    pred = upfirdn2d(data)
    pred.sync()
jt.sync_all(True)
end = time.time()

print("Jittor upfirdn2d cuda FPS:", (rerun * batch_size) / (end - start))