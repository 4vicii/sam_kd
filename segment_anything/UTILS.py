from Teacher_model import Teacher_model
from Student_model import StudentImageEncoderViT
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import List
import os
import logging

def setup_logging(output_dir):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(output_dir, 'training.log')),
                                  logging.StreamHandler()])
    return logging



class UnifiedTeacherStudent(nn.Module):
    def __init__(self, teacher_model: Teacher_model, student_model: StudentImageEncoderViT):
        super(UnifiedTeacherStudent, self).__init__()  # Important to call this if inheriting nn.Module
        self.teacher_model = teacher_model
        self.student_model = student_model
        # self.teacher_model.eval()
        # self.student_model.train()

    def forward(self, images: List[np.ndarray], image_format: str = "RGB") -> torch.Tensor:
        # Teacher's image processing
        with torch.no_grad():
            teacher_features, preprocessed_batch = self.teacher_model.set_images(images, image_format)

        # Forward these images through the student model (assuming student model has a forward method)
        student_output = self.student_model(preprocessed_batch)

        # If needed, you can return both teacher's and student's outputs
        return teacher_features, student_output


class DistillationMSENormalizedLoss(nn.Module):
    def __init__(self, student_feature_dim, teacher_feature_dim, resize_stu=True):
        super(DistillationMSENormalizedLoss, self).__init__()
        self.resize_stu = resize_stu
        # Define the adapter layer to project student's feature to teacher's feature dimension
        self.adapter = nn.Linear(student_feature_dim, teacher_feature_dim)

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances."""
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        # Adapt student features to match teacher's feature dimensions
        student_features = self.adapter(student_features)

        # Resize if necessary
        if student_features.shape[2:] != teacher_features.shape[2:]:
            if self.resize_stu:
                student_features = F.interpolate(student_features, teacher_features.shape[2:], mode='bilinear')
            else:
                teacher_features = F.interpolate(teacher_features, student_features.shape[2:], mode='bilinear')

        # Make sure shapes are same after potential resizing
        assert student_features.shape == teacher_features.shape

        # Normalize the features
        norm_student = self.norm(student_features)
        norm_teacher = self.norm(teacher_features)

        # Compute the MSE loss between normalized features
        loss = F.mse_loss(norm_student, norm_teacher) / 2

        return loss
