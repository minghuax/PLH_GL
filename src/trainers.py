import torch
import logging
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryAveragePrecision

class Trainer:
    def __init__(self, args, model, data, logger):
        self.args = args
        self.model = model
        self.d = data
        self.loss_fn = F.binary_cross_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=0)
        self.log_step = args.log.log_step
        self.logger = logger

    def train(self, n_epochs=2500, run=None):
        train_loss=0
        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            x = self.model(self.d.G_data, self.d.train_edges, self.d.train_x)
            loss = F.binary_cross_entropy(x, self.d.train_y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            val_loss, val_roc, val_acc = self.val()
            tmp_test_roc, tmp_test_acc = self.test()
            self.logger.add_result(run, [val_roc, val_acc, tmp_test_roc, tmp_test_acc])
            if (epoch+1) % self.log_step == 0:
                logging.info(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item()}, ROC: {tmp_test_roc}, AP: {tmp_test_acc}")

    def val(self):
        self.model.eval()
        y = self.model(self.d.G_data, self.d.val_edges, self.d.val_x)
        loss = F.binary_cross_entropy(y, self.d.val_y)
        roc = BinaryAUROC().to(self.model.device)
        ap = BinaryAveragePrecision().to(self.model.device)
        return loss, roc(y, self.d.val_y), ap(y, self.d.val_y.long())

    def test(self):
        self.model.eval()
        y = self.model(self.d.G_data, self.d.test_edges, self.d.test_x)
        roc = BinaryAUROC().to(self.model.device)
        ap = BinaryAveragePrecision().to(self.model.device)
        return roc(y, self.d.test_y), ap(y, self.d.test_y.long())