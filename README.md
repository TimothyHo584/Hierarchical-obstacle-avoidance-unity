# Hierarchical-obstacle-avoidance-unity
Simulate in unity environment.

## 環境版本需求
### PYTHON
* pytorch 1.10.2
* ml-agent 0.27.0 (依照unity官方ml-agents Release_18)
* cloudpickle 2.0.0
* (option) wandb 0.12.9
### Unity IDE Package
* Unity IDE 2019.4.30f1
* ML Agents 2.1.0-exp.1
* ML Agents Extensions 0.5.0-preview

## Install
### Step1 安裝Unity IDE
至Unity官網下載[unity hub](https://unity3d.com/cn/get-unity/download "Title")，並且選擇安裝2019.4.30f1版本。
### Step2 下載ML-agents release_18
[下載網址](https://github.com/Unity-Technologies/ml-agents/tree/release_18 "Title")
### Step3 安裝unity IDE所需的ML-agents套件
安裝`com.unity.ml-agents` & `com.unity.ml-agents.extensions`。

詳細教學請洽官方[DOCS](https://github.com/Unity-Technologies/ml-agents/blob/release_18/docs/Installation.md "Title")
### Step4 載入unitypackage
請前往[下載地址]()

~~ps.檔案太大無法放入專案內，請容小弟我放在外部空間。~~
### Step5 安裝所需python package
mlagents相關套件安裝方式請洽[官方教學文件](https://github.com/Unity-Technologies/ml-agents/blob/release_18/docs/Installation.md "Title")

Note:本專案使用GPU訓練，請確認是否已安裝相關的驅動程式。pytorch可使用指令`torch.cuda.is_available()`確認。
## How to use
### 測試python與Unity環境之溝通
`python unityEnv_test.py`

### 使用預先訓練之模型
使用預先訓練之模型:

`python sac_v2_Ray_unity.py --test --behavior 'straight' --env 'BuildGame'`

可透過修改`--behavior`名稱，來改變載入的模型。(straight、pedestrian、escape)

訓練模型:

`python sac_v2_Ray_unity.py --train --behavior 'straight' --env 'BuildGame'`

參數使用說明:
* `--wandb` 線上圖形化顯示即時的訓練狀態。(需自行安裝套件)
* `--env` 選擇執行編譯好的遊戲or直接使用IDE訓練。(BuildGame、Editor)
* `--run_id` 如果使用編譯遊戲，且須要多開訓練，請記得在執行程式前給予每個環境不同的編號。(如有相同號碼，系統會跳錯)
* `--turbo` 環境加速選項，可選擇加速倍率。