# Official Image-as-an-IMU integration

Source: [jerredchen/image-as-an-imu](https://github.com/jerredchen/image-as-an-imu),
Jerred Chen and Ronald Clark, ICCV 2025. The adapter was checked against the published
[`iaai/model.py`](https://github.com/jerredchen/image-as-an-imu/blob/main/iaai/model.py)
interface on 2026-09-07.

The project needs the **official pretrained SegNeXt checkpoint** for real image motion
extraction. It does not contain a newly trained imitation network. Follow the upstream
README for current dependency requirements and the checkpoint download. Do not commit
the checkpoint or a copy of the third-party repository here.

Example setup in a compatible GPU environment (install the upstream requirements as
directed in its README):

```bash
git clone https://github.com/jerredchen/image-as-an-imu.git third_party/image-as-an-imu
python -m pip install -r third_party/image-as-an-imu/requirements.txt
python -m pip install -e third_party/image-as-an-imu
python -m pip install -e .

python -m blur_gs prepare-motion --data data/my-scene/scene.json --checkpoint checkpoints/image-as-imu.pth --device cuda
```

The model is instantiated with `supervise_pose=False`, loaded strictly with the provided
state dictionary, set to eval, and frozen. BLUR-GS consumes its dense flow and depth heads.
The upstream pose head uses a mean focal length and hardcoded CUDA calls. Our separate
weighted least-squares implementation supports different fx/fy and principal points, and
can run on CPU. It uses GS depth for a scene-scale initialization; it is not an independent
learned motion estimator.

The upstream `infer` method center-crops to aspect ratio 320/224, resizes to 320x224,
and applies ImageNet normalization. We invoke that method, then restore its predictions
to the input grid. Flow x/y values are multiplied by the corresponding inverse resize
factors. Depth values are not multiplied by those image-scale factors. Cropped-out borders
receive zero confidence. The adapter requires the documented output tensor shapes and
fails explicitly if a later upstream version changes them.

Each cache records checkpoint SHA-256, source, flow reference, sign ambiguity, and confidence
provenance. The supplied initial version has no official checkpoint in the local workspace;
therefore no successful pretrained inference run is claimed. CPU tests cover restoration,
confidence/flow handling, the geometry solver, and downstream optimization. The upstream
network's dependencies and checkpoint must still be exercised on the training machine.

Important limitations inherited from the algorithm: single-image motion has a temporal
sign ambiguity; instantaneous velocity assumes an exposure duration and a small-motion
model; the paper's network was trained primarily on indoor data. No per-frame exposure
duration is guessed by this project.
