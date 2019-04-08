from leakage.trainer import Train
from itertools import count
from leakage.trainer import get_test_data_from_txt
import torch



trainer = Train()
# trainer.load_MNIST('/home/chen/MNIST/')
trainer.load_dataset('/home/chen/NM-dataset')
# trainer.learn()
# trainer.load_model('./models/model_299999.para')
#
# data = get_test_data_from_txt('a671.txt')
# print(data.shape)
#
# out = trainer.model(data.float()).cpu()
# print(torch.argmax(out))
#

# trainer.learn_MNIST()
#
for t in count():
    if t % 100000 == 99999:
        trainer.save_model('model_NM' + str(t) + '.para')
    trainer.learn()

