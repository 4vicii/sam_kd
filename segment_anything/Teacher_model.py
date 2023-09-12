# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


from segment_anything.modeling import Teacher_sam

from typing import Optional, Tuple, List

from .utils.transforms import ResizeLongestSide


class Teacher_model:
    def __init__(
        self,
        sam_model: Teacher_sam,
    ) -> None:
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    def set_images(
            self,
            images: List[np.ndarray],
            image_format: str = "RGB",
    ) -> torch.Tensor:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."

        input_images_torch_list = []

        for image in images:
            if image_format != self.model.image_format:
                image = image[..., ::-1]

            # Transform the image to the form expected by the model
            input_image = self.transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image, device=self.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous().unsqueeze(0)  # Make it BCHW
            input_images_torch_list.append(input_image_torch)

        batched_images_torch = torch.cat(input_images_torch_list, dim=0)

        return self.set_torch_images(batched_images_torch)

    @torch.no_grad()
    def set_torch_images(
            self,
            transformed_images: torch.Tensor,
    ) -> torch.Tensor:
        assert (
                len(transformed_images.shape) == 4
                and transformed_images.shape[1] == 3
        ), f"set_torch_images input must be BCHW."
        preprocessed_images_list = []
        for img in transformed_images:
            assert max(*img.shape[
                        -2:]) == self.model.image_encoder.img_size, \
                f"Each image in set_torch_images batch must have long side {self.model.image_encoder.img_size}."
            preprocessed_images_list.append(self.model.preprocess(img.unsqueeze(0)))

        # Concatenate the preprocessed images into one batch
        preprocessed_batch = torch.cat(preprocessed_images_list, dim=0)

        # Pass the preprocessed batch through the image encoder
        features = self.model(preprocessed_batch)
        return features, preprocessed_batch

    @property
    def device(self) -> torch.device:
        return self.model.device

