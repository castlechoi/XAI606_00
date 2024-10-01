from torch.utils.data import DataLoader

from benchmark.tasks.VocalSet.VocalSetT_dataset import (
    AudioDataset as VocalSetTAudioDataset,
)
from benchmark.tasks.VocalSet.VocalSetT_dataset import (
    FeatureDataset as VocalSetTFeatureDataset,
)
from benchmark.utils.config_utils import search_enumerate


def get_audio_datasets(args):
    train_sampler, valid_sampler, test_sampler = None, None, None
    train_collate_fn, valid_collate_fn, test_collate_fn = None, None, None

    Task_Dataset = eval(f"{args.dataset.dataset}AudioDataset")
    train_dataset = Task_Dataset(args, split="train")
    valid_dataset = Task_Dataset(args, split="valid")
    test_dataset = Task_Dataset(args, split="test")

    return (
        (train_dataset, train_sampler, train_collate_fn),
        (valid_dataset, valid_sampler, valid_collate_fn),
        (test_dataset, test_sampler, test_collate_fn),
    )


def get_feature_datasets(args):
    Task_Dataset = eval(f"{args.dataset.dataset}FeatureDataset")
    layer = search_enumerate(
        args.model.downstream_structure.components, name="feature_selector", key="layer"
    )
    train_dataset = Task_Dataset(
        feature_dir=args.dataset.input_dir,
        metadata_dir=args.dataset.metadata_dir,
        split="train",
        layer=layer,
    )
    valid_dataset = Task_Dataset(
        feature_dir=args.dataset.input_dir,
        metadata_dir=args.dataset.metadata_dir,
        split="valid",
        layer=layer,
    )
    test_dataset = Task_Dataset(
        feature_dir=args.dataset.input_dir,
        metadata_dir=args.dataset.metadata_dir,
        split="test",
        layer=layer,
    )

    train_collate_fn = train_dataset.train_collate_fn
    valid_collate_fn = valid_dataset.test_collate_fn
    test_collate_fn = test_dataset.test_collate_fn

    train_sampler, valid_sampler, test_sampler = None, None, None

    return (
        (train_dataset, train_sampler, train_collate_fn),
        (valid_dataset, valid_sampler, valid_collate_fn),
        (test_dataset, test_sampler, test_collate_fn),
    )


dataset_functions = {"feature": get_feature_datasets, "audio": get_audio_datasets}


def get_dataloaders(args):
    dataset_type = args.dataset.input_type

    if dataset_type in dataset_functions:
        (
            (train_dataset, train_sampler, train_collate_fn),
            (valid_dataset, valid_sampler, valid_collate_fn),
            (test_dataset, test_sampler, test_collate_fn),
        ) = dataset_functions[dataset_type](args)
    else:
        raise NotImplementedError(
            f"get_dataloaders() of dataset type {dataset_type} not implemented"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.dataloader.batch_size.train,
        shuffle=True,
        num_workers=args.dataloader.num_workers,
        collate_fn=train_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.dataloader.batch_size.valid,
        shuffle=False,
        num_workers=args.dataloader.num_workers,
        collate_fn=test_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.dataloader.batch_size.test,
        shuffle=False,
        num_workers=args.dataloader.num_workers,
        collate_fn=test_collate_fn,
    )

    return train_loader, valid_loader, test_loader
