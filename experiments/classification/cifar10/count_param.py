from torch_uncertainty.models.resnet import batched_resnet, ensemble_resnet, packed_resnet, resnet
from torch_uncertainty.models.wrappers.mc_dropout import mc_dropout

resnet18 = resnet(in_channels=3, num_classes=10, arch=18, style="cifar", dropout_rate=0.1)

models = {
    "Standard": resnet18,
    "BatchEnsemble": batched_resnet(
        in_channels=3, num_classes=10, arch=18, num_estimators=4, style="cifar"
    ),
    "PackedEnsemble": packed_resnet(
        in_channels=3, num_classes=10, arch=18, num_estimators=4, style="cifar", alpha=2, gamma=2
    ),
    "MCDropout": mc_dropout(
        model=resnet18,
        num_estimators=20,
    ),
    "DeepEnsembles": ensemble_resnet(
        in_channels=3, num_classes=10, arch=18, num_estimators=4, style="cifar"
    ),
}

for name, model in models.items():
    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {num_params}")
