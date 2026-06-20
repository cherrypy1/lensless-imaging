import hydra
import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    params = [(name, param) for name, param in model.named_parameters()]
    admm_params = [
        param
        for name, param in params
        if param.requires_grad and "params.values" in name
    ]
    other_params = [
        param
        for name, param in params
        if param.requires_grad and "params.values" not in name
    ]
    optimizer_config = {
        key: value
        for key, value in config.optimizer.items()
        if key not in ("_target_", "admm_lr")
    }
    admm_lr = config.optimizer.get("admm_lr", config.optimizer.lr)
    optimizer = get_class(config.optimizer._target_)(
        [
            {"params": other_params},
            {"params": admm_params, "lr": admm_lr},
        ],
        **optimizer_config,
    )
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
