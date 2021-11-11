# PicToAnime
A GAN that converts a photograph into a anime-esque image

## Demo Video

***maybe?***

<!-- ABOUT THE PROJECT -->
## About The Project

Using a standard GAN architecture, with a few tweaks such as a LeakyReLU and InstanceNorm, I created a GAN that takes a photograph as an input and outputs the photo but turned into an anime style image. I made use of WAndB to track training and Colab's free GPU.



Here are some resources I used to implement PicToAnime:

* [WhiteBox Cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization)
* [Dataset](https://github.com/TachibanaYoshino/AnimeGAN/releases/download/dataset-1/dataset.zip)
* [AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN)
 

Results: 
Input: ![](https://user-images.githubusercontent.com/36611240/141239005-a690aba1-d400-483b-9ca3-8bf619f2941a.png)

Output: ![](https://user-images.githubusercontent.com/36611240/141239039-7f7de7ee-fc3f-4969-b578-7d8bad30f43c.png)

Input: ![](https://user-images.githubusercontent.com/36611240/141239079-e5db155d-758d-4737-8d15-29a456e2f377.png)

Output: ![](https://user-images.githubusercontent.com/36611240/141239106-3b734e57-acac-4c8f-bfdc-a545fa47ab36.png)




### Built With

* [PyTorch](https://pytorch.org)
* [NumPy](https://www.numpy.org)
* [PIL](https://pillow.readthedocs.io/)
* [WAndB](https://wandb.ai)
* [OpenCV](https://opencv.org)
* [MatPlotLib](https://matplotlib.org)


<!-- GETTING STARTED -->
## Getting Started

* *Clone git repository*
* *Make sure correct packages are installed(below)*
* *Run train.py*
* *or optionally for easier overall usage run PicToAnime.ipynb in Google Colaboratory or Jupyter Notebook*

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* numpy:
  ```pip install numpy```
* pytorch:
  ```pip install torch```
* PIL:
  ```pip install pillow```
* WAndB:
  ```pip install wandb```
* OpenCV
  ```pip install opencv-python```
* MatPlotLib
 ```pip install matplotlib```

See the [open issues](https://github.com/NNDEV1/PicToAnime/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Nalin Nagar - nalinnagar1@gmail.com


