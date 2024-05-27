## Description
Convert bvh files to smplx parameters, with mesh and joint renderer.
The camera is fixed to center of human now, sometimes it looks a little strange.

## Installation
### PLEASE FOLLOW THE INSTRUCTION OF PyTorch3D TO INSTALL TORCH AND PyTorch3D!!!
```
pip install bvh, scipy, pickle
```

## Fast start

Convert and render mesh:
```
python visualize.py -p ./demo/sample00_p1_fps20_Ske24013101001.bvh --render_mesh
```

Convert and render joint:
```
python visualize.py -p ./demo/sample00_p1_fps20_Ske24013101001.bvh
```
