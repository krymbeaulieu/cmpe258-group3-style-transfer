{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f176b69-afcb-4f96-8018-75b0df3190f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# style_classifier.py\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torchvision import models\n",
    "\n",
    "class StyleClassifier(nn.Module):\n",
    "    \"\"\"\n",
    "    ResNet50-based multi-class style classifier.\n",
    "    Fine-tune using train_style_classifier.py.\n",
    "    \"\"\"\n",
    "    def __init__(self, num_styles):\n",
    "        super().__init__()\n",
    "        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)\n",
    "        self.model.fc = nn.Linear(self.model.fc.in_features, num_styles)\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.model(x)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
