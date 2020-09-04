# PyTorch FGVC Dataset

This repo contains some unofficial PyTorch dataset APIs(mainly for Fine-Grained Visual Categorization task), 
which support automatically download (except large-scale datasets), extract the archives, and prepare the data.

## Supported Datasets
- [x] CUB-200-2011
- [x] Stanford Dogs
- [x] Stanford Cars
- [x] FGVC Aircraft
- [x] NABirds
- [x] Tiny ImageNet
- [x] iNaturalist 2017
- [ ] Oxford 102 Flowers
- [ ] Oxford-IIIT Pets
- [ ] Food-101

## Usage
The code was tested on 
- pytorch==1.4.0
- torchvision==0.4.1

Use them the similar way you use `torchvision.datasets`.
```python
train_dataset = Cub2011('./cub2011', train=True, download=False)
test_dataset = Cub2011('./cub2011', train=False, download=False)
```
## Contributing
Feel free to open an issue or PR.

## License
MIT