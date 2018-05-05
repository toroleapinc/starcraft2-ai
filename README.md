# SC2 RL Agent

Reinforcement learning agent for StarCraft II mini-games using PySC2. Implements A2C with a CNN policy that reads spatial features from the screen/minimap.

Tested on MoveToBeacon and CollectMineralShards.

```
pip install -r requirements.txt
python train.py --map MoveToBeacon --episodes 3000
```

Requires StarCraft II installed + PySC2.
