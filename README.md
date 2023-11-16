# [Identification of highly boosted H → γγ decays with the ATLAS detector using deep neural networks](https://cds.cern.ch/record/2878576?ln=en)

## Overview

This thesis presents the development and implementation of two innovative jet tagging algorithms designed for the identification of highly boosted H → γγ decays using the ATLAS detector at the Large Hadron Collider (LHC). These algorithms leverage advanced neural network architectures to enhance the identification process. The code for this work is contained in this repository.

## Algorithms

### Deep Neural Network (DNN) Based Tagger

- **Performance**: Comparable to existing algorithms designed for highly boosted Z → e+e− decays.
- **Features**:
  - Multifunctional, effective in also identifying Z → e+e− decays.
  - High classification rates, comparable to [another DNN jet tagging algorithm](https://cds.cern.ch/record/2845238) for highly boosted heavy bosons

### Adversarial Neural Network (ANN) Based Tagger

- **Architecture**: Utilizes an Adversarial Neural Network for mass-decorrelated classification.
- **Noteable results**:
  - Very slight performance decrease compared to DNN-based tagger.
  - Also functional as a dual-use jet tagger.
  - Achieves a 27.8% reduction in mutual information between the mass feature and scalar discriminant metric.
  - Demonstrates enhanced rejection rates for background (τ)τ-jets.

```
@thesis{Hey:2878576,
      author        = "Hey, Nathaniel",
      title         = "{Identification of highly boosted H → γγ decays with
                       the ATLAS detector using deep neural networks}",
      school        = "University of Edinburgh",
      year          = "2023",
      url           = "https://cds.cern.ch/record/2878576",
      note          = "Presented 29 Sep 2023",
}
```

