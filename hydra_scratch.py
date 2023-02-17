from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from src.utils.utils import instantiate_callbacks, instantiate_loggers
from src.utils.utils import get_pylogger

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    print(cfg)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    model = instantiate(cfg.model)
    dm = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)
    logger = instantiate_loggers(cfg.get('logger'))
    print("hi")

    
    

if __name__ == "__main__":
    main()