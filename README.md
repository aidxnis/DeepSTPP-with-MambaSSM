# DeepSTPP with MambaSSM 🌌

![DeepSTPP with MambaSSM](https://img.shields.io/badge/DeepSTPP%20with%20MambaSSM-v1.0-blue)

Welcome to the **DeepSTPP with MambaSSM** repository! This project enhances the Deep Spatiotemporal Point Process (DeepSTPP) model by integrating the Mamba State Space Model. This integration allows for improved modeling of spatiotemporal data, enabling more accurate predictions and insights in various applications.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Topics](#topics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Releases](#releases)

## Introduction

The DeepSTPP model is a powerful framework for understanding and predicting events in spatiotemporal contexts. By incorporating the Mamba State Space Model, this version enhances the model's ability to capture complex patterns in data that vary over space and time. 

The model is particularly useful in fields such as urban planning, epidemiology, and environmental science, where understanding the dynamics of events in space and time is crucial.

## Features

- **Encoder-Decoder Architecture**: Utilizes an encoder-decoder structure to capture temporal dependencies.
- **Hawkes Process Integration**: Models self-exciting point processes for better event prediction.
- **Mamba State Space Model**: Enhances state estimation and prediction accuracy.
- **Recurrent Neural Networks**: Leverages RNNs for capturing long-term dependencies in data.
- **Spatiotemporal Analysis**: Supports analysis of data that varies across both space and time.
- **Structured State Space Models**: Offers flexibility in modeling complex relationships.

## Installation

To install the DeepSTPP with MambaSSM, clone the repository and install the required packages.

```bash
git clone https://github.com/aidxnis/DeepSTPP-with-MambaSSM.git
cd DeepSTPP-with-MambaSSM
pip install -r requirements.txt
```

Ensure you have Python 3.7 or higher installed. 

## Usage

After installation, you can run the model using the following command:

```bash
python main.py --config config.yaml
```

Make sure to adjust the `config.yaml` file to suit your dataset and modeling requirements. 

## Model Architecture

The architecture of DeepSTPP with MambaSSM consists of several key components:

1. **Input Layer**: Accepts spatiotemporal data.
2. **Encoder**: Processes input sequences to capture temporal dependencies.
3. **Decoder**: Generates predictions based on encoded representations.
4. **Mamba State Space Module**: Enhances state estimation, allowing for better predictions over time.
5. **Output Layer**: Produces the final predictions.

![Model Architecture](https://miro.medium.com/max/1400/1*I0HCV7W8vG8G5Z2w0u1BZw.png)

## Topics

This repository covers a range of topics relevant to the model:

- **Encoder-Decoder**: Techniques for processing sequences.
- **Hawkes Process**: A model for self-exciting events.
- **Mamba**: A state space model that improves prediction accuracy.
- **Neural Spatiotemporal Point Process**: Neural networks applied to point process modeling.
- **Recurrent Marked Temporal Point Process**: A method for modeling events with marks.
- **Spatial AI**: Artificial intelligence applications in spatial analysis.
- **Spatiotemporal Analysis**: Techniques for analyzing data that varies over space and time.
- **Transformer**: A model architecture that has gained popularity in various domains.

## Contributing

We welcome contributions from the community. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Create a pull request.

Please ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback, please reach out via email at [your-email@example.com].

## Releases

To download the latest version of DeepSTPP with MambaSSM, visit the [Releases section](https://github.com/aidxnis/DeepSTPP-with-MambaSSM/releases). Here you can find all the versions, including the latest updates and improvements.

Feel free to explore the repository and contribute to the ongoing development of this exciting project!