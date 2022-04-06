import os
from argparse import ArgumentParser
from asyncio.unix_events import BaseChildWatcher
from pytorch_lightning import Trainer
from network import WSOLSystem
from dataset import CUB200Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

def train(args):
    # dict_args = vars(args)
    if args.algorithm_name.lower() == "wsol":
        cur_dataset = CUB200Dataset(batch_size=args.batch_size)
        cur_model = WSOLSystem(backbone=args.backbone,
                               mea_m_dim=args.mea_m_dim,
                               class_num=args.class_num,
                               lr=args.lr,
                               num_workers=args.num_workers,
                               max_epoch=args.max_epochs,
                               lmbda1=args.lmbda1,
                               lmbda2=args.lmbda2,
                               lmbda3=args.lmbda3,
                               lmbda4=args.lmbda4,
                               )

    else:
        raise KeyError("Algorithm Name: {} is not supported".format(
            temp_args.algorithm_name))
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        filename="wsol-cub200-{epoch:02d}-{val_loss:.2f}",
    )
    args.callbacks = [checkpoint_callback]
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(cur_model, cur_dataset)


def inference(args):
    if args.algorithm_name.lower() == "wsol":
        cur_dataset = CUB200Dataset()
        cur_model = WSOLSystem(backbone=args.backbone, class_num=args.class_num)
    else:
        raise KeyError("Algorithm Name: {} is not supported".format(
            temp_args.algorithm_name))
    trainer = Trainer.from_argparse_args(args)
    print("######### start to predict #########")
    trainer.predict(cur_model, cur_dataset, ckpt_path=args.infer_ckpt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument("--algorithm_name", type=str,
                        default="wsol", help="wsol", choices=['wsol'])
    parser.add_argument("--mode", type=str,
                        default="train", help="train/infer", choices=['train', 'infer'])
    parser.add_argument("--infer_ckpt", type=str, default="",
                        help="model for inference", required=False)
    temp_args, _ = parser.parse_known_args()

    if temp_args.algorithm_name == "wsol":
        parser = WSOLSystem.add_model_specific_args(parser)
    else:
        raise KeyError("Algorithm Name: {} is not supported".format(
            temp_args.algorithm_name))

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        assert os.path.isfile(args.infer_ckpt) != "", f"inference model: {args.infer_ckpt} is not a valid path"
        inference(args)
