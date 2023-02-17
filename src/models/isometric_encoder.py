from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.optim import Optimizer
import torch
from torchmetrics.functional import pairwise_euclidean_distance

# for now, will train to measure how locally isometric it gets
class LightningIsometricEncoder(LightningModule):
    def __init__(self, encoder: nn.Module, optimizer: Optimizer, isometry_measure: float):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.isometry_measure = isometry_measure


    def forward(self, x):
        return self.encoder.forward(x)

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())

    def training_step(self, batch, batch_idx):
        x, idx = batch[0], batch[1].tolist()
        z = self.forward(x)
        loss, Jt_J = self._isometry_loss(self.encoder, x)

        return {"embedding": z, "loss": loss, "Jt_J": Jt_J}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.print(f"Batch Loss: {outputs['loss']}")
        if batch_idx % 10 == 0:
            self.print(outputs["Jt_J"])

    def on_train_end(self):
        X = torch.from_numpy(self.trainer.datamodule.numpy_data).float().unique(dim=0)

        distances = pairwise_euclidean_distance(X.view(X.size(0), -1))
        distances = torch.where(distances == 0.0, 1e9, distances)
        smallest_5_percent = torch.quantile(distances.flatten(), self.isometry_measure, interpolation='nearest')
        indicies = (distances < smallest_5_percent).nonzero().tolist()
        i = [idx[0] for idx in indicies]
        j = [idx[1] for idx in indicies]
        X_distances = distances[i, j]

        encodings = self(X.permute(0,-1,-2,-3))
        distances_encodings = pairwise_euclidean_distance(encodings)
        correspondnig_distance_encodings = distances_encodings[i, j]
        final_result = torch.stack((X_distances, correspondnig_distance_encodings), dim=1).permute(1, 0)
        corr = torch.corrcoef(final_result)
        print("train end")
        print(corr[0, 1])


    def _isometry_loss(self, E, x, create_graph=True):
        flattend_x = x.view(x.size(0), -1)
        
        def _func_sum(flattend_x):
            x = flattend_x.view(flattend_x.size(0), *self.trainer.datamodule.shape[1:])
            return E(x).view(-1, 1).sum(dim=0)
        J = torch.autograd.functional.jacobian(_func_sum, flattend_x, create_graph=create_graph).permute(1,0,2)
        J = J.view(J.size(0), J.size(1), x.size(-2), x.size(-1)).squeeze()
        Jt = torch.transpose(J, -2, -1)
        Jt_J = torch.matmul(Jt, J)
        I = torch.stack(
            [
                torch.eye(self.trainer.datamodule.shape[-1]) for _ in range(flattend_x.size(0))
            ]
        )
        loss = torch.norm((Jt_J - I), p='fro')**2
        return loss, Jt_J