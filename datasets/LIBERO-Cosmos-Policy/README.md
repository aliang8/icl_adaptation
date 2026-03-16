---
license: cc
---
# LIBERO-Cosmos-Policy

## Dataset Description

LIBERO-Cosmos-Policy is a modified version of the [LIBERO simulation benchmark dataset](https://github.com/Lifelong-Robot-Learning/LIBERO), created as part of the Cosmos Policy project. This is the dataset used to train the [Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) checkpoint.

### Key Modifications

Our modifications include the following:

1. **Higher-resolution images**: Images are saved at 256×256 pixels (vs. 128×128 in the original).
2. **No-op actions filtering**: Transitions with "no-op" (zero) actions that don't change the robot's state are filtered out.
3. **JPEG compression**: Images are JPEG-compressed to reduce storage requirements.
4. **Deterministic regeneration**: All demonstrations are replayed in the simulation environment with deterministic seeding for reproducibility.

### Dataset Structure

The dataset is organized into two main directories:

- **`success_only/`**: Contains only successful demonstration episodes (filtered version). These are demonstrations that succeeded when replayed in the simulation environments (it turns out that some demonstrations fail due to human errors during teleoperation). This set is used to train Cosmos Policy to generate high-quality actions.
- **`all_episodes/`**: Contains all episodes, including both successful and failed demonstrations. This set is used to train Cosmos Policy's world model and value function.

Each directory contains data from four task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
- `libero_spatial` - Tasks testing spatial reasoning (10 tasks, 50 demos each)
- `libero_object` - Tasks testing object manipulation (10 tasks, 50 demos each)
- `libero_goal` - Tasks testing language-specified goals (10 tasks, 50 demos each)
- `libero_10` - Long-horizon manipulation tasks (10 tasks, 50 demos each)

### Data Format

Each HDF5 file in `success_only/` contains:
```
data/
├── demo_0/
│   ├── obs/
│   │   ├── agentview_rgb_jpeg     # Third-person camera images (JPEG compressed)
│   │   ├── eye_in_hand_rgb_jpeg   # Wrist camera images (JPEG compressed)
│   │   ├── gripper_states         # Gripper joint positions
│   │   ├── joint_states           # Robot joint positions
│   │   ├── ee_states              # End-effector states (position + orientation)
│   │   ├── ee_pos                 # End-effector position
│   │   └── ee_ori                 # End-effector orientation
│   ├── actions                     # Action sequence
│   ├── states                      # Environment states
│   ├── robot_states                # Combined robot state (gripper + EEF pos + EEF quat)
│   ├── rewards                     # Sparse rewards (0 until final timestep)
│   └── dones                       # Episode termination flags
├── demo_1/
...
```

The `all_episodes/` directory contains rollout data in a different format. Each episode is stored as a separate HDF5 file with the naming pattern:
```
episode_data--suite={suite_name}--{timestamp}--task={task_id}--ep={episode_num}--success={True/False}--regen_demo.hdf5
```

Each of these HDF5 files contains:
```
# Datasets (arrays)
primary_images_jpeg      # Third-person camera images (JPEG compressed), shape: (T, H, W, 3)
wrist_images_jpeg        # Wrist camera images (JPEG compressed), shape: (T, H, W, 3)
proprio                  # Proprioceptive state (gripper + EEF pos + quat), shape: (T, 9)
actions                  # Action sequence, shape: (T, 7)

# Attributes (scalars/metadata)
success                  # Boolean: True if episode succeeded, False otherwise
task_description         # String: Natural language task description
```

### Statistics

- **Total demonstrations**: 500 per task suite before filtering (4 suites = 2000 total)
- **Success rate**: ~80-90% (varies by task suite)
- **Image resolution**: 256×256×3 (RGB)
- **Action dimensions**: 7 (6-DoF end-effector control + 1 gripper)
- **Proprioception dimensions**: 9 (2 gripper joints + 3 EEF position + 4 EEF quaternion)

### Original LIBERO Dataset

This dataset is derived from the original LIBERO benchmark:
- **Paper**: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
- **Repository**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **License**: CC BY 4.0

### Citation

If you use this dataset, please cite both the original LIBERO paper and the Cosmos Policy paper:
<!-- ```bibtex
@article{liu2023libero,
  title={Libero: Benchmarking knowledge transfer for lifelong robot learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={44776--44791},
  year={2023}
}

# TODO: Add Cosmos Policy BibTeX
``` -->

### License

Creative Commons Attribution 4.0 International (CC BY 4.0)