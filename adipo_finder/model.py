import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class AdipoNN(nn.Module):
    """
    Simple feed-forward neural network for adipocyte classification.
    """

    def __init__(self, input_dim: int):
        super(AdipoNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdipoModel:
    """
    Class to handle model training, evaluation, and prediction.
    """

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seed for reproducibility.
        """
        random.seed(seed)  # Python random
        np.random.seed(seed)  # NumPy
        torch.manual_seed(seed)  # PyTorch CPU
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # PyTorch GPU
            torch.cuda.manual_seed_all(seed)  # if using multi-GPU
            torch.backends.cudnn.deterministic = True  # make CUDA deterministic
            torch.backends.cudnn.benchmark = False  # disable auto-optimization

    @staticmethod
    def get_y_from_df(df: pd.DataFrame) -> torch.Tensor:
        y = (df["ground_truth"] > 0).to_numpy().astype(np.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return y

    @staticmethod
    def get_training_input_from_df(
        df: pd.DataFrame,
    ) -> tuple[torch.Tensor, StandardScaler]:
        """
        Prepare data for training and return the scaler.
        """
        # scale them
        X = df.drop(columns=["segment_id", "ground_truth", "image_id"]).values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        return X, scaler

    @staticmethod
    def get_prediction_input_from_df(
        df: pd.DataFrame, scaler: StandardScaler
    ) -> torch.Tensor:
        """
        Prepare data for prediction using an existing scaler.
        """
        X = df.drop(columns=["segment_id", "ground_truth", "image_id"]).values
        X = scaler.transform(X)  # only transform, don’t fit
        X = torch.tensor(X, dtype=torch.float32)
        return X

    @classmethod
    def train_model(
        cls,
        full_df: pd.DataFrame,
        n_epochs: int = 1000,
        seed: int = 42,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
    ) -> tuple[AdipoNN, list, list, list, StandardScaler]:
        """
        Train the neural network model.
        """
        cls.set_seed(seed)
        image_ids = full_df["image_id"].unique()
        train_ids, test_ids = train_test_split(
            image_ids, test_size=test_frac, random_state=42
        )  # don't use seed here, better to always get the same
        train_ids, val_ids = train_test_split(
            train_ids, test_size=val_frac, random_state=42
        )  # split
        print(
            f"Training: {len(train_ids)}, validation: {len(val_ids)}, test: {len(test_ids)}"
        )

        train_mask = full_df["image_id"].isin(train_ids)
        val_mask = full_df["image_id"].isin(val_ids)
        full_df["image_id"].isin(test_ids)

        # we scale the data by fitting a scaler to the training set. We then save
        # the scaler so we can use the same scaler later when predicting. We use
        # the scaler on the validation and test sets
        X_train, scaler = cls.get_training_input_from_df(full_df.loc[train_mask])
        y_train = cls.get_y_from_df(full_df.loc[train_mask])

        X_val = cls.get_prediction_input_from_df(full_df.loc[val_mask], scaler)
        y_val = cls.get_y_from_df(full_df.loc[val_mask])

        # Create datasets and loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            worker_init_fn=lambda _: np.random.seed(seed),
        )
        val_loader = DataLoader(val_dataset, batch_size=32)

        model = AdipoNN(X_train.shape[1])

        # Loss and optimizer
        num_pos = (y_train == 1).sum()
        num_neg = (y_train == 0).sum()
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        # Training loop
        for epoch in range(n_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
            val_loss /= len(val_loader)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss:.4f}")

        return model, train_ids, val_ids, test_ids, scaler

    @classmethod
    def evaluate_model(
        cls,
        model: AdipoNN,
        scaler: StandardScaler,
        full_df: pd.DataFrame,
        test_ids: list,
    ) -> None:
        """
        Evaluate the model on the test set.
        """
        # Make sure your model is in evaluation mode
        model.eval()
        test_mask = full_df["image_id"].isin(test_ids)

        X_test = cls.get_prediction_input_from_df(full_df.loc[test_mask], scaler)
        y_test = cls.get_y_from_df(full_df.loc[test_mask])

        # Disable gradient computation for evaluation
        with torch.no_grad():
            y_pred_prob = model(X_test)  # outputs between 0 and 1
            y_pred = (y_pred_prob > 0.5).int()  # convert to 0 or 1

        # Convert tensors back to numpy for sklearn metrics
        y_pred_np = y_pred.numpy()
        y_test_np = y_test.numpy()

        # Classification report
        print(classification_report(y_test_np, y_pred_np))

    @staticmethod
    def save_model(
        filename: str, model: AdipoNN, train_ids: list, val_ids: list, test_ids: list
    ) -> None:
        """
        Save the model and split IDs to a file.
        """
        input_dim = model.net[0].in_features
        torch.save(
            {
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
            filename,
        )

    @staticmethod
    def load_model(filename: str) -> tuple[AdipoNN, list, list, list]:
        """
        Load a saved model.
        """
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        model = AdipoNN(input_dim=checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return (
            model,
            checkpoint["train_ids"],
            checkpoint["val_ids"],
            checkpoint["test_ids"],
        )

    @classmethod
    def predict_and_clean_image(
        cls,
        model: AdipoNN,
        scaler: StandardScaler,
        samp_id: str,
        full_df: pd.DataFrame,
        unfiltered_seg: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict adipocytes on a single image and return a cleaned segmentation mask.
        """
        df_samp = full_df.loc[full_df["image_id"] == samp_id]
        X_samp = cls.get_prediction_input_from_df(df_samp, scaler)
        segmented_sample = unfiltered_seg
        model.eval()
        with torch.no_grad():
            y_pred_prob = torch.sigmoid(model(X_samp)).numpy().flatten()
            y_pred = (y_pred_prob > threshold).astype(bool)

        segment_ids = df_samp["segment_id"].values
        segment_ids_to_rem = segment_ids[~y_pred]

        cleaned_image = np.copy(segmented_sample)
        mask = np.isin(cleaned_image, segment_ids_to_rem)
        cleaned_image[mask] = 0
        return cleaned_image
