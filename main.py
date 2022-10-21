import torch
import argparse
from os import path
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from models import ConvNext, SWIN
from data import cifar10, isic_2019

def main():
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["convnext", "swin"], type=str, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "isic_2019"], type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--job_id", type=str, default=None)
    args = parser.parse_args()

    # print(args.model)
    # print(args.dataset)
    # print(args.epochs)
    # print(args.batch_size)
    # print(args.lr)

    main_path = path.join("experiments", args.model + ":" + args.dataset, args.job_id)
    chp_path = path.join(main_path, "checkpoints")
    log_path = path.join(main_path, "logs")

    print("creating checkpoint.")
    checkpoint_callback = ModelCheckpoint(dirpath=chp_path,
                                          filename="{epoch}-{val_loss:.4f}",
                                          monitor="val_loss",
                                          save_top_k=3,
                                          mode="min")

    if args.model == "convnext":
        print("initializing ConvNext model.")
        model = ConvNext.ConvNext(lr=args.lr)
        trainer = ConvNext.get_convnext_trainer(gpus=args.gpus,
                                                max_epochs=args.epochs,
                                                callbacks=[checkpoint_callback],
                                                log_path=log_path)
        cifar10_image_size = (128, 128)
        isic_2019_image_size = (224, 224)
    else:
        print("initializing SWIN model.")
        model = SWIN.SWIN(lr=args.lr)
        trainer = SWIN.get_swin_trainer(gpus=args.gpus,
                                        max_epochs=args.epochs,
                                        callbacks=[checkpoint_callback],
                                        log_path=log_path)
        cifar10_image_size = (224, 224)
        isic_2019_image_size = (256, 256)

    if args.dataset == "cifar10":
        print("preparing CIFAR10 dataset.")
        feature_extractor = cifar10.get_cifar10_feature_extractor(cifar10_image_size)
        train_dataset, test_dataset = cifar10.get_cifar10_data(
            root=args.data_path,
            train_transforms=feature_extractor,
            test_transforms=feature_extractor
        )
    else:
        print("preparing ISIC 2019 dataset.")
        feature_extractor = isic_2019.get_isic_2019_feature_extractor(image_size=isic_2019_image_size)
        train_dataset, test_dataset = isic_2019.get_isic_2019_data(
            root=args.data_path,
            transform=feature_extractor
        )

    print("creating data loaders.")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers)

    print("fitting model.")
    trainer.fit(model,
                train_dataloader,
                test_dataloader)


if __name__ == '__main__':
    main()
