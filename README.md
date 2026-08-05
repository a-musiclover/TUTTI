# TUTTI

**TUTTI** is a deep learning system for audio to score (A2S), converting audio recordings directly into symbolic music notation in [ABC format](https://abcnotation.com/).

## Overview

Given an audio recording as input, it generates a structured ABC notation transcription as output — enabling downstream applications such as score editing, music analysis, and retrieval.

## Status

> 🚧 This repository is a work in progress. The pretrain data, full code and model weights will be released upon paper publication.
> I promise I will update everything (especially the data) ASAP!)

## Requirements

```bash
pip install -r requirements.txt
```

## Citation

Our paper has been accepted by **ISMIR 2026**! 

The official camera-ready paper and arXiv preprint will be available soon. For now, if you use **TUTTI**, our dataset, or code in your research, please cite us using the following BibTeX entry:

```bibtex
@inproceedings{hu2026tutti,
  title     = {TUTTI: Toward Generalizable Audio-to-Score Transcription via Fully Synthesized Data},
  author    = {Hu, Jianhuai and Wang, Yashan and Wu, Shangda and Guo, Zhancheng and Liang, Shijie and Meng, Wuna and Yang, Chuanqi and Li, Xiaobing and Yu, Feng and Sun, Maosong},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  note      = {Accepted for publication},
  url       = {[https://github.com/a-musiclover/TUTTI](https://github.com/a-musiclover/TUTTI)}
}
```

## License

* **Code**: Standard [MIT License](LICENSE).
* **Dataset (TuttiCorpus)**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Non-Commercial use only).
