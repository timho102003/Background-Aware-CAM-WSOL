from curses import meta
from pickletools import optimize
import xxlimited
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models.resnet as res_dict


def fc_init_weight(m):
    torch.nn.init.xavier_normal_(m.weight)
    torch.nn.init.zeros_(m.bias)

def zero_init_weight(m):
    torch.nn.init.zeros_(m.weight)
    torch.nn.init.zeros_(m.bias)

class ResNetBase(torch.nn.Module):
    def __init__(self, backbone: str) -> None:
        super(ResNetBase, self).__init__()
        body = getattr(res_dict, backbone)(pretrained=True)
        self.conv1 = body.conv1
        self.bn1 = body.bn1
        self.relu = body.relu
        self.maxpool = body.maxpool
        self.layer1 = body.layer1
        self.layer2 = body.layer2
        self.layer3 = body.layer3
        self.layer4 = body.layer4
        self.in_features = body.fc.in_features
        del body

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MutualExclusiveAggregator(torch.nn.Module):
    def __init__(self, feature_dim: int, m: int = 60) -> None:
        super(MutualExclusiveAggregator, self).__init__()
        self.m = m
        self.feature_dim = feature_dim
        self.mea_layer = torch.nn.Conv2d(
            self.feature_dim, self.m, kernel_size=1)
        self.mea_layer.apply(zero_init_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        mea_out = self.mea_layer(input)
        mea_out = mea_out.reshape(b, self.m, -1)
        mea_out = torch.nn.Softmax(2)(mea_out)
        input = input.reshape(b, c, -1)
        output = torch.einsum("ijk,ilk->ijl", [input, mea_out])
        # output = torch.matmul(input, mea_out.permute(0, 2, 1).contiguous())
        output = torch.mean(output, dim=2)
        return output


class Classifier(torch.nn.Module):
    def __init__(self, class_num: int, bottleneck_dim: int = 256) -> None:
        super(Classifier, self).__init__()
        self.fc = torch.nn.utils.weight_norm(
            torch.nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(fc_init_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.fc(input)
        return output


class WSOLSystem(pl.LightningModule):
    def __init__(self, backbone: str = 'resnet18', mea_m_dim: int = 60, class_num: int = 200, lr: float = 1.7e1-4, num_workers: int = 0, max_epoch: int = 50,
                 lmbda1: float = 1.0, lmbda2: float = 0.3, lmbda3: float = 0.3, lmbda4: float = 0.2) -> None:
        super(WSOLSystem, self).__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.backbone = ResNetBase(backbone=backbone)
        self.mea_o = MutualExclusiveAggregator(
            feature_dim=self.backbone.in_features, m=mea_m_dim)
        self.mea_b = MutualExclusiveAggregator(
            feature_dim=self.backbone.in_features, m=mea_m_dim)
        self.cls_o = Classifier(class_num=class_num,
                                bottleneck_dim=self.backbone.in_features)
        self.cls_b = Classifier(class_num=class_num,
                                bottleneck_dim=self.backbone.in_features)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.multi_label_soft_margin_loss = torch.nn.MultiLabelSoftMarginLoss()
        self.feature_blobs = []

    def on_predict_start(self) -> None:
        print("setting params object/background")
        param_object = list(self.cls_o.parameters())
        param_background = list(self.cls_b.parameters())
        self.weight_softmax_o = np.squeeze(param_object[-1].data)
        self.weight_softmax_b = np.squeeze(param_background[-1].data)
        def _feature_hook(module, input, output):
            self.feature_blobs.append(output)
        print("finish register feature hook")

        self.backbone._modules.get(
            "layer4").register_forward_hook(_feature_hook)
        self.predict_outpath = "./cam_out"
        if not os.path.isdir(self.predict_outpath):
            os.makedirs(self.predict_outpath)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WSOL")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=0)
        # parser.add_argument("--seed", type=int, default=2020, help="random seed")
        parser.add_argument("--lr", type=float,
                            default=1.7e-04, help="learning rate")
        # Model Configuration
        parser.add_argument("--backbone", type=str,
                            default="resnet101", help="resnet backbone")
        parser.add_argument("--mea_m_dim", type=int,
                            default=60, help="m dimension in MEA module")
        parser.add_argument("--class_num", type=int,
                            default=200, help="class number")
        parser.add_argument("--lmbda1", type=int,
                            default=1, help="obj obj ratio")
        parser.add_argument("--lmbda2", type=int,
                            default=0.3, help="bg obj ratio")
        parser.add_argument("--lmbda3", type=int,
                            default=0.3, help="obj bg ratio")
        parser.add_argument("--lmbda4", type=int,
                            default=0.2, help="bg bg ratio")
        return parent_parser

    def cal_stagger_loss(self, oo, ob, bo, bb, targets):
        # tar_oo = torch.zeros(oo.size()).scatter_(
        #     1, targets.cpu(), 1).to(oo.device)
        tar_bo = torch.ones(bo.size()).scatter_(1, targets.cpu(), 0).to(bo.device)
        tar_ob = torch.zeros(ob.size(), device=ob.device)
        tar_bb = torch.ones(bb.size(), device=bb.device)
        loss_oo = self.ce_loss(oo, targets.squeeze(1))
        loss_bo = self.multi_label_soft_margin_loss(bo, tar_bo)
        loss_ob = self.multi_label_soft_margin_loss(ob, tar_ob)
        loss_bb = self.multi_label_soft_margin_loss(bb, tar_bb)

        stagger_classifier_loss = self.hparams.lmbda1 * loss_oo + \
            self.hparams.lmbda2 * loss_bo + \
            self.hparams.lmbda3 * loss_ob + \
            self.hparams.lmbda4 * loss_bb
        return {"loss": stagger_classifier_loss, "l_oo": loss_oo, "l_bo": loss_bo, "l_ob": loss_ob, "l_bb": loss_bb}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-04, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[15, 30, 45], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def step(self, batch, batch_idx):
        x, y, _ = batch
        out = self.backbone(x)
        zo = self.mea_o(out)
        zb = self.mea_b(out)
        oo = self.cls_o(zo)
        bo = self.cls_b(zo)
        ob = self.cls_o(zb)
        bb = self.cls_b(zb)
        logs = self.cal_stagger_loss(oo, ob, bo, bb, targets=y)
        with torch.no_grad():
            h_x = F.softmax(oo, dim=-1).data.squeeze()
            probs, idx = h_x.sort(-1, True)
            acc = (idx[:, 0] == y.squeeze(1)).float().mean()
            logs.update({"accuracy": acc.item()})
        return logs["loss"], logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()},
                      on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_meta_{k}": v for k, v in logs.items()},
                      on_step=True, on_epoch=False)
        self.log('val_loss', loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()},
                      on_step=True, on_epoch=False)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, meta_info = batch
        imgName = meta_info["filename"][0]
        ori_h = meta_info["size"][0].item()
        ori_w = meta_info["size"][1].item()
        logit = self.cls_o(self.mea_o(self.backbone(x)))
        h_x = F.softmax(logit, -1).data.squeeze()
        if not len(h_x.squeeze().shape):
            h_x = h_x.unsqueeze(0)
        probs, idx = h_x.sort(-1, True)
        cam_dict = self.generate_cam(self.feature_blobs[0], [
                                     idx[0].item()], outsize=(ori_h, ori_w))
        img_show = cv2.imread(imgName)
        heatmap_mix = cv2.applyColorMap(
            cam_dict["mix"][0], colormap=cv2.COLORMAP_JET)
        heatmap_bg = cv2.applyColorMap(
            cam_dict["bg"][0], colormap=cv2.COLORMAP_JET)
        heatmap_fg = cv2.applyColorMap(
            cam_dict["obj"][0], colormap=cv2.COLORMAP_JET)
        result_mix = heatmap_mix * 0.5 + img_show * 0.5
        result_bg = heatmap_bg * 0.5 + img_show * 0.5
        result_fg = heatmap_fg * 0.5 + img_show * 0.5
        result_cam = np.concatenate(
            [img_show, result_mix, result_fg, result_bg], axis=1)
        class_dir = os.path.join(self.predict_outpath,
                                 imgName.split(os.sep)[-2])
        if not os.path.isdir(class_dir):
            os.makedirs(class_dir)
        cv2.imwrite(os.path.join(
            class_dir, imgName.split(os.sep)[-1]), result_cam)
        self.feature_blobs = []  # init feature_blobs

    def generate_cam(self, feature_conv, class_idx, outsize):
        
        def _norm_and_resize(img, size):
            img = img - torch.min(img)
            img[img < 0] = 0
            cam_img = img / (torch.max(img) + 1e-06)
            cam_img = np.uint8(255 * cam_img.cpu().numpy())
            cam_img = cv2.resize(cam_img, (size[1], size[0]))
            return cam_img

        bs, nc, h, w = feature_conv.shape
        cam_dict = {"mix": [], "obj": [], "bg": []}
        class_idx = [class_idx]
        cam_obj = torch.matmul(
            self.weight_softmax_o[class_idx], feature_conv.reshape((nc, h * w)))
        cam_bg = torch.matmul(
                self.weight_softmax_b[class_idx], feature_conv.reshape((nc, h * w)))
        cam_mix = torch.concat([cam_bg, cam_obj])
        cam_idx = torch.argmax(cam_mix, dim=0)
        cam = torch.where(cam_idx == 1, cam_obj, torch.zeros_like(cam_obj))
        cam_dict["mix"].append(_norm_and_resize(cam.reshape(h, w), outsize))
        cam_dict["obj"].append(_norm_and_resize(
            cam_obj.reshape(h, w), outsize))
        cam_dict["bg"].append(_norm_and_resize(cam_bg.reshape(h, w), outsize))
        return cam_dict
