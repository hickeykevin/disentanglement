import pytorch_lightning as pl
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import pickle

class SklearnLightningModule(pl.LightningModule):
    def __init__(self, methods: List[Dict], save_dir: str):
        super().__init__()
        self.methods = methods
        self.save_dir = Path(save_dir)

    def on_train_start(self) -> None:
        self.final_embeddings = {}
        data = self.trainer.datamodule.numpy_data.reshape(
            self.trainer.datamodule.numpy_data.shape[0], 
            -1
            )
        self.print("[INFO] Training all Sklearn Models")
        for method in tqdm(self.methods):
            name = str(method)
            embedding = method.fit_transform(data)
            self.final_embeddings[name] = embedding

    def on_train_end(self) -> None:
        with open(self.save_dir / "embeddings.pkl", "wb") as file:
            pickle.dump(self.final_embeddings, file)
        self.print(f"Dumped embeddings to {self.save_dir}")
    
    def configure_optimizers(self) -> Any:
        pass

    def training_step(self, batch, batch_idx) -> Any:
        pass

