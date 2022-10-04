import argparse
from models import ConvNext, SWIN
from data import cifar10, isic_2019
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["convnex", "swin"], type=str, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "isic_2019"], type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("gpus", type=int, default=1)
    args = parser.parse_args()

    print(args.model)
    print(args.dataset)
    print(args.epochs)
    print(args.batch_size)
    print(args.lr)

    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_path,
                                          filename="{epoch}-{val_loss:.4f}",
                                          monitor="val_loss",
                                          save_top_k=3,
                                          mode="min")

    if args.model == "convnex":
        model = ConvNext.ConvNext()
        trainer = ConvNext.get_convnext_trainer(gpus=args.gpus,
                                                max_epochs=args.epochs,
                                                callbacks=[checkpoint_callback])
        cifar10_image_size = (128, 128)
    else:
        model = SWIN.SWIN()
        trainer = SWIN.get_swin_trainer(gpus=args.gpus,
                                        max_epochs=args.epochs,
                                        callbacks=[checkpoint_callback])
        cifar10_image_size = (224, 224)

    if args.dataset == "cifar10":
        feature_extractor = cifar10.get_cifar10_feature_extractor(cifar10_image_size)
        train_dataset, test_dataset = cifar10.get_cifar10_data(
            root="data/datasets",
            train_transforms=feature_extractor,
            test_transforms=feature_extractor
        )
    else:
        feature_extractor = isic_2019.get_isic_2019_feature_extractor(image_size=cifar10_image_size)
        train_dataset, test_dataset = isic_2019.get_isic_2019_data(
            root="data/datasets",
            transform=feature_extractor  
        )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size)

    trainer.fit(model, train_dataloader, test_dataloader)

if __name__ == '__main__':
    main()
