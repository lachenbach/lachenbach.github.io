---
title: Latent Video Models in 2026
description: Understanding current state and emerging trends.
date: 2026-07-20
draft: true
---


## Which video models are we going over in this post
| Model | Open Weights | Release Date | Capabilities | Link |
|---|---|---|---|---|
| Cosmos-predict-1 | Yes | .. | .. | https://arxiv.org/pdf/2501.03575 |
| Cosmos-Predict-2.5 | Yes | .. | .. | https://arxiv.org/pdf/2511.00062 |
| Cosmos-3 | Yes | .. | .. | https://arxiv.org/pdf/2606.02800 |
| Wan2.1/2.2 | Yes | .. | .. | https://arxiv.org/pdf/2503.20314 |
| Huanyuan 1.5 | Yes | .. | .. | https://arxiv.org/abs/2511.18870 |
| Seedance 2.0 | No | .. | .. | https://arxiv.org/pdf/2604.14148 |
| LTX 2/2.3 | Yes | .. | .. | https://arxiv.org/pdf/2601.03233 |
| Helios | ? | .. |
| MagI-1 | ? | .. |
| Skyreels-V2 | ? | .. |
| Veo 3.1 | No | .. |
| OpenSora | Yes | .. |

## Data
Modern technical reports on video models disclose a fair amount about their data preprocessing pipelines but rarely disclose their data sources.

| Model | Pretraining dataset size images | Pretraining dataset size video |
|---|---|---|---|
| Huanyuan 1.5 |  |5B | 800 million video segments? |

### Preprocessing Videodata
Constructing a pretraining video dataset naturally involves challenges unique to the modality. Since modern video model training often starts with training an image-generation model, data preparation also has to be done for the image datasets. For now, we're just touching on video preprocessing steps.

**Shot detection:** One video might show multiple completely different shots (scene change in a movie), which need to be split into seperate clips. Shot detection is a well studied problem and learned approaches like TransNetV2 seem to perform sufficiently well.

**Transcoding:** The videos from your large corpus will probably have different formats (H.264, H.265, AV1, etc.), GOP structures, bitrates, etc. One would align these into one format to avoid processing errors in the preprocessing pipeline. The importance of choosing GPUs with the right accelerators (L40S has both NVDEC and NVENC, while H100 only has NVDEC) is well described in the cosmos-predict-1 paper.

**Visual Quality Filtering:** Removing video with static scenes, abrupt camera movement can be achieved by training small classifiers/scorers [1, 5]. Clustering the dataset before training the scorer avoids scoring long tails of the dataset as "poor" [5].  

**Motion Quality Filtering:** Removing blur, motion jitter, noise and other artifacts can also be done using lightweight classifiers trained on human-rated videos like DOVER. One can further classify videos into more static scenes (landscape, interviews, etc.) and reduce their sampling ratio during training [5].

**Additional Filtering:** Cosmos-predict series, focused on physical understanding, removes videos with text overlay. When building a "physically plausible video model", it's also advisable to remove any video game footage etc. Semantic artifact filtering like VTSS can also be applied. One may use a VLM to futher remove unwanted clips. Remove black/padding borders (ffmpeg cropdetect), grid layouts, etc. [4]. Naturally, too short video clips are filtered. Synthetic video detection and removal.

**Annotation:** DALL-E3 shows that visual generation quality can be improved by highky descriptive captions [5]. The VLMs used as captioning models are asked to provide extensive captions which describe the semantics of the scene and movements involved to achieve that effect. Prompts can be produced in different lengths, which will enable different use cases downstream [1]. One challenge in captioning is, that efforts to write more detailed prompts for the video model can lead to an increased hallucination rate [4]. There's an interesting OPA-DPO approach in [4], however, it's not immediately clear how effective it is. The only benefit I see with training your own captioning model as done in [5] is the camera angle prediction.

**Deduplication:** One can use video embeddings (like InternVideo2, Cosmos-Embed1-448p) and perform deduplication DataComp-style [1,3]. Due to the scale, one will need to cluster and then identify the near-duplicates. A library to do these searches is Milvus used in [2].

**Sampling/curriculum:** The highest quality video data should naturally be retained for the later stages of training. In Wan2.1 post-training they select this data as a mixture of the highest scores given by all the quality classifiers and human manual judgement. Cosmos-3 additionally injects synthetic images and videos into the mid-training mix. Their synthetic datasets target simulated physics, robots, driving, humans?, and warehouse footage. 



## Going from Video to Latents


## Architectures


## Training 





[1] cosmos predict 1
[2] cosmos predict 2.5
[3] cosmos-3
[4] huanyuan-video 1.5
[5] wan2.1/2.2