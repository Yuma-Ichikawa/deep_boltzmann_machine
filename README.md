# Deep Boltzmann Machine(DBM)の実装 (Implementation of Deep Boltzmann Machine)

## 概要(Overview)
Pytorchを使用しDBMの実装を行った

## 内容(Contents)
- DBMは学習に時間がかかるためPytorchを使用しGPUに対応させた
- 事前学習として層ごとの貪欲学習をRBMを使用し行った
- 事前学習のRBMの学習はPersistent Contrastive Divergence(PCD)法を使用した
- Positive Partの評価に平均場近似を使用した
- Negative Partの評価にPersistent Contrastive Divergence(PCD)法を使用した
- 学習時間のLogを確認可能

## 使用したモジュール(Modules)
- numpy
- pytorch
- scipy
- time
