import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.dataloader.batch_size = 1

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    save_path = ROOT_PATH / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    skip_model_load = config.inferencer.get("from_pretrained") is None
    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=None,
        skip_model_load=skip_model_load,
    )
    inferencer.run_inference()
    print(save_path)


if __name__ == "__main__":
    main()
