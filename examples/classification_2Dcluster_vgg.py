# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import glob
import argparse
import sklearn.metrics as metrics
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import MinkowskiEngine as ME

from torch.utils.data import Dataset, DataLoader

from examples.common import seed_all

parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", type=float, default=0.05)
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--val_freq", type=int, default=5)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--stat_freq", type=int, default=10)
parser.add_argument("--weights", type=str, default="modelnet.pth")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--translation", type=float, default=0.2)
parser.add_argument("--test_translation", type=float, default=0.0)
parser.add_argument(
    "--network",
    type=str,
    choices=["pointnet", "minkpointnet", "minkfcnn"],
    default="minkfcnn",
)

class clusterCNN(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
            channels = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512),
        D=2,
    ):
        ME.MinkowskiNetwork.__init__(self, D)

        self.network_initialization(
            in_channel,
            out_channel,
            channels = channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiReLU(),
        )
    def network_initialization(
        self, in_channel, out_channel, channels, embedding_channel, kernel_size, D=2,
    ):
        self.conv1_1 = self.get_conv_block(in_channel, channels[0], kernel_size=kernel_size, stride=1)
        self.conv1_2 = self.get_conv_block(channels[0], channels[1], kernel_size=kernel_size, stride=1)
        self.pool1 = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.conv2_1 = self.get_conv_block(channels[1], channels[2], kernel_size=kernel_size, stride=1)
        self.conv2_2 = self.get_conv_block(channels[2], channels[3], kernel_size=kernel_size, stride=1)
        self.pool2 = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.conv3_1 = self.get_conv_block(channels[3], channels[4], kernel_size=kernel_size, stride=1)
        self.conv3_2 = self.get_conv_block(channels[4], channels[5], kernel_size=kernel_size, stride=1)
        self.conv3_3 = self.get_conv_block(channels[5], channels[6], kernel_size=kernel_size, stride=1)
        self.pool3 = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.conv4_1 = self.get_conv_block(channels[6], channels[7], kernel_size=kernel_size, stride=1)
        self.conv4_2 = self.get_conv_block(channels[7], channels[8], kernel_size=kernel_size, stride=1)
        self.conv4_3 = self.get_conv_block(channels[8], channels[9], kernel_size=kernel_size, stride=1)
        self.pool4 = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.conv5_1 = self.get_conv_block(channels[9], channels[10], kernel_size=kernel_size, stride=1)
        self.conv5_2 = self.get_conv_block(channels[10], channels[11], kernel_size=kernel_size, stride=1)
        self.conv5_3 = self.get_conv_block(channels[11], channels[12], kernel_size=kernel_size, stride=1)
        self.pool5 = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.global_pool = ME.MinkowskiGlobalPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(512, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            self.get_mlp_block(512, 2),
            #ME.MinkowskiFunctional.softmax(),
        )
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
    
    def forward(self, x: ME.SparseTensor):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.global_pool(x)
        return self.final(x).F



class nnbarDataset(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "/data/yjwa/sparse_torch/MinkowskiEngine/nnbar_overlay",
    ):
        Dataset.__init__(self)
        self.w_xy, self.w_val, self.label = self.load_data(data_root, phase)
        self.phase = phase

    def load_data(self, data_root, phase):
        #print("In load_data")
        w_xy, w_val, labels = [], [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, "*_%s*.npy" % phase))
        assert len(files) > 0, "No files found"
        for npy_name in files:
            #print(npy_name)
            a=np.load(npy_name, allow_pickle=True)
            #with h5py.File(h5_name) as f:
            #print("f")
            w_xy.extend(a[:,10:12])
            w_val.extend(a[:,12])
            #print("data extent")
            labels.extend(a[:,20].astype("int64"))
            #print("labels extent")

        w_xy = np.stack(w_xy, axis=0)
        #w_val = np.stack(w_val, axis=0)
        #print(np.shape(w_xy), np.shape(w_val), np.shape(labels))
        labels = np.stack(labels, axis=0)
        
        return w_xy, w_val, labels

    def __getitem__(self, i: int) -> dict:
        w_xy = self.w_xy[i]
        w_val = self.w_val[i]
        #print(w_xy)
        
        #if self.phase == "train":
        #    np.random.shuffle(w_xy)
        label = self.label[i]
        w_val = torch.from_numpy(np.asarray(w_val))
        #print("in getitem")

        #print("before padding")

        #print(w_xy[0])
        #print(w_xy[1])

        #print("after padding")

        padding = np.asarray(np.zeros((4096,))).astype("int64")
        #print(padding)
        w_x = np.array(w_xy[0])
        w_x = np.concatenate((w_x,padding))
        w_x = w_x[:4096]
        #print(w_x)

        #np.reshape(1,2,)

        w_y = np.array(w_xy[1])
        w_y = np.concatenate((w_y,padding))
        w_y = w_y[:4096]
        #print(w_y)
        w_xy = np.dstack((w_x,w_y))
        w_xy = np.reshape(w_xy, (4096,2))
        #print(w_xy)
        #print(w_xy.shape)
        w_xy = torch.from_numpy(np.asarray(w_xy))
        
        label = np.reshape(label, (1))

        w_val = np.concatenate((w_val, padding))
        w_val = w_val[:4096]
        w_val = np.reshape(w_val, (4096,1))
        #print(w_val.shape)

        w_val = torch.from_numpy(np.asarray(w_val))
        label = torch.from_numpy(np.asarray(label))
#        return {
#            "coordinates": w_xy.to(torch.float32),
#            "feats": w_val.to(torch.float32),
#            "label": label,
#        }
        return w_xy.to(torch.float32), w_val.to(torch.float32), label

    def __len__(self):
        return self.w_xy.shape[0]




def make_data_loader_custom(phase, config):
    assert phase in ["train", "val", "test"]
    is_train = phase == "train"
    dataset = nnbarDataset(
        phase=phase,
    )
    #print(dataset)
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        shuffle = is_train,
        batch_size=config.batch_size,
        collate_fn=ME.utils.batch_sparse_collate,
    )


def criterion(pred, labels, smoothing=True):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    labels = labels.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, labels, reduction="mean")

    return loss


def val(net, device, config, phase="val"):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    data_loader = make_data_loader_custom("val", config=config,)

    net.eval()
    labels_val, preds_val = [], []
    with torch.no_grad():
        for batch in data_loader:
            coords, feats, labels = batch
            input = ME.SparseTensor(feats.float(), coords, device="cuda")
            logit = net(input)
            pred = torch.argmax(logit, 1)

            #print("val_labels", labels)
            #print("val_logit", logit)
            #print("val_pred", pred)
            #labels.append(labels.numpy())
            labels_val.append(labels.cpu().numpy())
            preds_val.append(pred.cpu().numpy())
            torch.cuda.empty_cache()
    return metrics.accuracy_score(np.concatenate(labels_val), np.concatenate(preds_val))

def test(net, device, config, phase="test"):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    data_loader = make_data_loader_custom("test", config=config,)

    net.eval()
    labels_val, preds_val, sfs_val = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            coords, feats, labels = batch
            input = ME.SparseTensor(feats.float(), coords, device="cuda")
            logit = net(input)
            sf_val = torch.softmax(logit, dim=1)
            pred = torch.argmax(logit, 1)

            labels_val.append(labels.cpu().numpy())
            preds_val.append(pred.cpu().numpy())
            sfs_val.append(sf_val.cpu().numpy())
            torch.cuda.empty_cache()

    return np.concatenate(labels_val), np.concatenate(preds_val), np.concatenate(sfs_val)
    #return metrics.accuracy_score(np.concatenate(labels_val), np.concatenate(preds_val))



def train(net, device, config):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    optimizer = optim.SGD(
        net.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps,)
    print(optimizer)
    print(scheduler)

    train_iter = iter(make_data_loader_custom("train", config))
    best_metric = 0
    net.train()
    for i in range(config.max_steps):
        optimizer.zero_grad()
        try:
            data = train_iter.next()
        except StopIteration:
            train_iter = iter(make_data_loader_custom("train", config))
            data = train_iter.next()

        coords, feats, labels = data
        input = ME.SparseTensor(feats.float(), coords, device="cuda")
        logit = net(input)
        loss = criterion(logit, labels.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if i % config.stat_freq == 0:
            print(f"Iter: {i}, Loss: {loss.item():.3e}")

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.weights,
            )
            accuracy = val(net, device, config, phase="val")
            if best_metric < accuracy:
                best_metric = accuracy
            print(f"Validation accuracy: {accuracy}. Best accuracy: {best_metric}")
            net.train()


if __name__ == "__main__":
    config = parser.parse_args()
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# changed "cuda" to "cuda:0"
    print("===================ModelNet40 Dataset===================")
    print(f"Training with translation {config.translation}")
    print(f"Evaluating with translation {config.test_translation}")
    print("=============================================\n\n")

    net = clusterCNN(
        in_channel=1, out_channel=2,).to(device)
    print("===================Network===================")
    print(net)
    print("=============================================\n\n")

    train(net, device, config)
    y_label, y_pred, y_sf = test(net, device, config, phase="test")
    accuracy = metrics.accuracy_score(y_label, y_pred)
    print(f"Test accuracy: {accuracy}")

    dim0 = len(y_label)
    y_label = np.reshape(y_label,(dim0,1))
    y_pred = np.reshape(y_pred,(dim0,1))
    y_sf = np.reshape(y_sf,(dim0,2))
    y_combine = np.hstack((y_label, y_pred, y_sf))
    #y_pred_sf = torch.softmax(torch.from_numpy(y_pred), dim=2)
    np.savetxt('label_pred_sf_w_plane.csv',(y_combine), delimiter=' ')
        #numpy.savetxt("y_pred_sf.csv", y_pred_sf, delimiter=",")
    #accuracy = test(net, device, config, phase="test")

