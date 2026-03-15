# In-Context Imitation Learning via Next-Token Prediction
by <a href="https://max-fu.github.io">Max (Letian) Fu*</a>, <a href="https://qingh097.github.io/">Huang Huang*</a>, <a href="https://www.linkedin.com/in/gaurav-datta/">Gaurav Datta*</a>, <a href="https://yunliangchen.github.io/">Lawrence Yunliang Chen</a>, <a href="https://autolab.berkeley.edu/people">William Chung-Ho Panitch</a>, <a href="https://fangchenliu.github.io/">Fangchen Liu</a>, <a href="https://www.research.autodesk.com/people/hui-li/">Hui Li</a>, and <a href="https://goldberg.berkeley.edu">Ken Goldberg</a> at UC Berkeley and Autodesk (*equal contribution).

[[Paper](https://icrt.dev/files/icrt.pdf)] | [[Project Page](https://icrt.dev/)] | [[Checkpoints](https://huggingface.co/mlfu7/ICRT)] | [[Dataset](https://huggingface.co/datasets/Ravenh97/ICRT-MT)] | [[Citation](#citation)]

This repo contains the checkpoints for *In-Context Imitation Learning via Next-Token Prediction*. We investigate how to bring few-shot, in-context learning capability that exists in next-token prediction models (i.e. GPT) into real-robot imitation learning policies. 

In particular, we store the pre-trained vision encoder and ICRT model separately. Please find them in [encoder](crossmae_rtx/cross-mae-rtx-vitb.pth), [ICRT](icrt_vitb_droid_pretrained/icrt_vitb_droid_pretrained.pth), and [ICRT-Llama7B](icrt_llama7b_lora/icrt_llama7b_lora.pth).

Please refer to the [code](https://github.com/Max-Fu/icrt) on installing the repo, training and inferencing the model. 

## Dataset Structure

```
ICRT-MT
├── merged_data_part1.hdf5
│   ├── episode_1
│   │   ├── observation
│   │       ├── exterior_image_1_left
│   │       └── exterior_image_2_left
│   │       └── wrist_image_left
│   │       └── cartesian_position
│   │       └── gripper_position
│   │       └── joint_position
│   │   ├── action
│   │       ├── cartesian_velocity
│   │       └── gripper_velocity
│   │       └── joint_velocity
│   │       └── cartesian_position
│   │       └── gripper_position
│   │       └── joint_position
│   │   ├── language_instruction
│   │   ├── language_instruction_2
│   │   ├── language_instruction_3
│   │   ├── language_embedding
│   │   ├── language_embedding_2
│   │   ├── language_embedding_3
│   │   ...
│   ├── episode_2
│   │   ...
│   └── episode_3
│       ...
└── merged_data_part1_keys.json
...
```

## Citation
Please give us a star 🌟 on Github to support us!

Please cite our work if you find our work inspiring or use our code in your work:
```
@article{fu2024icrt,
    title={In-Context Imitation Learning via Next-Token Prediction}, 
    author={Letian Fu and Huang Huang and Gaurav Datta and Lawrence Yunliang Chen and William Chung-Ho Panitch and Fangchen Liu and Hui Li and Ken Goldberg},
    journal={arXiv preprint arXiv:2408.15980},
    year={2024}
}
```