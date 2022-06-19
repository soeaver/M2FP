# M2FP: Mask2Former for Parsing

By **Lu Yang**, **Hongjie Liu**, **Wenhe Jia**, **Qing Song**

`arXiv`(need to be updated)

<div align="center">
  <img src="https://bowenc0221.github.io/images/maskformerv2_teaser.png" width="100%" height="100%"/>
</div><br/>

### Features
* A single architecture for single human parsing, multiple (instance-level) human parsing, video human parsing and face parsing.
* Support several parsing datasets: LIP, ATR, PASCAL-Person-Part, CIHP, MHPv2, VIP, LaPa, CelebAMask-HQ.

## Updates
[2022/6/19] Code initialization.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for M2FP](datasets/README.md).

See [Getting Started with M2FP](GETTING_STARTED.md).

## Advanced usage

See [Advanced Usage of M2FP](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [M2FP Model Zoo](MODEL_ZOO.md).

## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

The majority of Mask2Former is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="CitingMask2Former"></a>Citing Mask2Former

If you use M2FP in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@article{yang2022parsing,
  title={Deep Learning Technique for Human Parsing: A Survey and Outlook},
  author={Lu Yang and Hongjie Liu and Wenhe Jia and Qing Song},
  journal={arXiv},
  year={2022}
}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@article{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={arXiv},
  year={2021}
}
```

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).
