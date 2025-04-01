#! /usr/bin/python3

import torch
import torchaudio
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms.v2 as VT
import torchaudio.transforms as AT
import lightning as L
from torchmetrics.classification import Accuracy, ConfusionMatrix
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import torch.nn as nn
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torch.utils.data as data
import random
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WavDataSet(Dataset):
    def __init__(self, stage='train', transform=None, target_transform=None, predict_dir=None, predict_script=None):
        self.transform=transform
        self.stage = stage
        self.predict_script = predict_script
        self.target_transform = target_transform
        self.df = pd.DataFrame()
        self.data_dir = '/local/202510_csci581_project/project_data/task2/'
        self.train_dir = self.data_dir + 'train/'
        self.dev_dir = self.data_dir + 'dev/'
        if stage=='predict':
            assert predict_dir is not None, "Must include a prediction directory while predicting"
            assert predict_script is not None, "Must point to a prediction script while predicting"
            self.wav_dir = predict_dir
            self.df = self.read_script(self.df)
        elif (stage=='dev'):
            self.wav_dir = self.dev_dir
            self.df = self.add_wav_files(self.df)
        else: ## default to train
            self.wav_dir = self.train_dir
            self.df = self.add_wav_files(self.df)
        self.waveforms = self.load_waveforms()
        if stage != 'predict':
            self.df = self.add_dialect(self.df)
        else: 
            self.df['dialect'] = None

    def read_script(self, df):
        with open(self.predict_script, 'r') as file:
            wav_paths = file.read().splitlines()
        df['wav'] = wav_paths
        return df

    def add_wav_files(self, df):
        wav_files = [f for f in os.listdir(self.wav_dir) if f.endswith(".wav")]
        df['wav'] = wav_files 
        return df

    def add_dialect(self, df):
        df["dialect"] = None
        for wav in df['wav']:
            dialect_file = wav.removesuffix(".wav") + ".dialect.txt"
            with open(os.path.join(self.wav_dir, dialect_file), 'r') as file:
                dialect = int(file.readline().strip())
                df.loc[df['wav'] == wav, 'dialect'] = dialect
        return df

    def load_waveforms(self):
        waveforms = []
        total = len(self.df)
        print("Loading waveforms from:", self.wav_dir)
        for idx in range(len(self.df)):
            print(f'{idx}/{total}', end='\r')
            wav = self.df.at[idx, 'wav']
            wav_path = os.path.join(self.wav_dir, wav)
            waveform, sample_rate = torchaudio.load(wav_path)
            waveforms.append(waveform)
        assert(len(waveforms) == len(self.df))
        return waveforms

    def get_class_counts(self):
        dialect_count = self.df.groupby('dialect').agg(count=('dialect', 'count')) 
        return dialect_count['count'].to_numpy(dtype=np.float64)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        assert self.waveforms is not None
        dialect = self.df.at[index, 'dialect']
        waveform = self.waveforms[index]
        if self.transform:
            transform_waveform = self.transform(waveform)
        if self.target_transform:
            dialect = self.target_transform(dialect)
        if self.stage == 'predict':
            return transform_waveform
        else:
            return transform_waveform, dialect

class WavDataModule(L.LightningDataModule):
    def __init__(self, sample_rate, batch_size, n_mels, num_workers, predict_dir=None, predict_script=None):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_mels = n_mels
        self.predict_dir = predict_dir
        self.predict_script = predict_script

        MelSpectrogram = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            n_mels=self.n_mels,
            win_length=400, # 25ms
            hop_length=160, #10ms
        )

        randspeed = VT.Lambda(lambda waveform: AT.Speed(16000, round(random.uniform(0.8,1.2), 2))(waveform)[0])
        InjectNoise = VT.Lambda(lambda waveform: waveform + torch.randn_like(waveform) * 0.0005)

        self.AugmentWaveform = VT.Compose([
            randspeed,
            InjectNoise
        ])
    
        self.toSpectrogram = VT.Compose([
            MelSpectrogram,
            AT.AmplitudeToDB(top_db=80), 
        ])

        mean, std = self.get_spectrogram_stats()
        self.train_transform = VT.Compose([
            self.AugmentWaveform,
            self.toSpectrogram,
            VT.Normalize([mean],[std]),
            VT.RandomCrop((n_mels, 400), pad_if_needed=True)
        ])

        self.test_transform = VT.Compose([
            self.toSpectrogram,
            VT.Normalize([mean],[std]),
            VT.CenterCrop((n_mels, 400))
        ])

    # Find mean and std of spectrograms without data augmentation
    def get_spectrogram_stats(self):
        spectrograms = WavDataSet(stage='train', transform=self.toSpectrogram)
        sum_mean = 0
        sum_std = 0
        for idx in range(len(spectrograms)):
            sum_mean += spectrograms[idx][0].mean()
            sum_std += spectrograms[idx][0].std()
        mean = sum_mean / len(spectrograms)
        std = sum_std / len(spectrograms)
        return mean, std
            
    def setup(self, stage=None):
        if stage == "fit":
            self.train = WavDataSet(
               stage='train', transform=self.train_transform, target_transform=None)

            ''' Weight Sampling'''   
            # weights = 1. / self.train.get_class_counts()
            # sample_weights = torch.zeros(len(self.train))
            # for i in range(len(self.train)):
            #     x,y = self.train[i]
            #     sample_weights[i] = weights[y]
            # self.weighted_rand_sampler = WeightedRandomSampler(sample_weights, len(self.train))
        
            self.dev = WavDataSet(
               stage='dev', transform=self.test_transform, target_transform=None)

        if stage == "test" or stage is None:
            print("Tried to call setup: test")
            exit()

        if stage == 'predict':
            self.predict = WavDataSet(
                stage='predict', transform=self.test_transform, target_transform=None, 
                predict_dir=self.predict_dir, predict_script=self.predict_script)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)
                        #   sampler=self.weighted_rand_sampler)

    def val_dataloader(self):
        return DataLoader(self.dev, shuffle=False, batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    #def test_dataloader(self):
    #    return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
    def predict_dataloader(self):
        return DataLoader(self.predict, shuffle=False, batch_size=self.batch_size, 
                          num_workers=self.num_workers)

def parse_all_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs [default : 10]')
    parser.add_argument('-lr', type=float, default=0.01,
                        help='Learning rate [default: 0.01]') 
    parser.add_argument('--batchsz', type=int, default=32,
                        help="Batch size [default: 32]")
    parser.add_argument('--nmels', type=int, default=60,
                        help="Number of mel filterbanks [default: 60]")
    parser.add_argument('--predict-dir', dest='predict_dir', type=str,
                        help="Base directory for predictions")
    parser.add_argument('--predict-script', dest='predict_script', type=str,
                        help="Script of filenames of files to predict on")

    return parser.parse_args()

class CNN(L.LightningModule):
    def __init__(self, lr, epochs, class_counts, n_mels):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = 8
        self.lr = lr
        self.epochs = epochs
        self.class_counts = class_counts
        self.n_mels = n_mels

        self.acc_metric = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.confusion_val = ConfusionMatrix(task='multiclass', num_classes=self.num_classes, normalize='true')
        self.confusion_train = ConfusionMatrix(task='multiclass', num_classes=self.num_classes, normalize='true')

        conv_block = lambda in_channels, out_channels, kernel_size: nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv_blocks = nn.Sequential(
            conv_block(1,64, 5),
            conv_block(64,64, 5),
            nn.MaxPool2d((2,3), (2,3)),
            conv_block(64, 128, 3),
            conv_block(128, 128, 3),
            nn.MaxPool2d((2,3), (2,3)),
            conv_block(128, 256, 3),
            conv_block(256, 256, 3),
            conv_block(256, 256, 3),
            nn.MaxPool2d((2,3), (2,3)),
        )
        
        self.fc1 = nn.Linear(6144, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.dropout = nn.Dropout(0.5)

        self.initialize_weights()
        FreqMasking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20, iid_masks=True)
        TimeMasking = torchaudio.transforms.TimeMasking(time_mask_param=30, iid_masks=True)
        self.AugmentSpectrogram = VT.Compose([
            FreqMasking,
            FreqMasking,
        ])
            
    def initialize_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0.01)
    

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def eval_batch(self, batch, stage=None):
        x, y = batch
        if (stage=='train'):
            x = self.AugmentSpectrogram(x)
            VT.RandomCrop((self.n_mels, 400), pad_if_needed=True)
        y_pred = self(x)
        weight=None
        if(stage=='train'):
            self.confusion_train.update(y_pred, y)
            if (self.class_counts is not None):
                weight = 1 / self.class_counts
                weight = weight.type_as(x)
        if (stage=='val'):
            self.confusion_val.update(y_pred, y)
        loss = F.cross_entropy(y_pred, y, weight=weight)
        acc = self.acc_metric(y_pred, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.eval_batch(batch, stage='train')
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.eval_batch(batch, stage='val')
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.eval_batch(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def on_validation_epoch_start(self):
        self.confusion_val.reset()

    def on_train_epoch_start(self):
        self.confusion_train.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return F.softmax(self(batch), dim=0).cpu().numpy().astype(np.float32)
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        return optim

    def plot_confusion_matrix(self):
        self.confusion_val.plot()
        plt.savefig("cm_val.png")
        plt.figure()
        self.confusion_train.plot()
        plt.savefig("cm_train.png")
        plt.close()
    
def main():
    args = parse_all_args()

    data = WavDataModule(16000, args.batchsz, args.nmels, num_workers=10, 
                         predict_dir=args.predict_dir, predict_script=args.predict_script)
    train_class_counts = torch.tensor([350., 700., 640., 470., 550., 260., 580., 180])
    if args.predict_dir:
        model = CNN.load_from_checkpoint('./best-model.ckpt')
        trainer = L.Trainer(accelerator='auto')
        predictions = trainer.predict(model, data)
        np.save("task2_predictions.npy", predictions)
        exit()

    model = CNN(lr=args.lr, epochs=args.epochs, class_counts=train_class_counts, n_mels=args.nmels)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",        
        mode="max",               
        filename="best-model-{epoch:02d}",
        save_top_k=1,            
        verbose=True
    )

    trainer = L.Trainer(max_epochs=args.epochs, accelerator="auto", callbacks = [checkpoint_callback])
    trainer.fit(model, data)

    model.plot_confusion_matrix()
    plt.savefig("cm1.png") 

if __name__ == "__main__":
    main()
