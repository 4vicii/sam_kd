import torch
from torch.utils.data import DataLoader
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from Student_model import StudentImageEncoderViT
from UTILS import DistillationMSENormalizedLoss, UnifiedTeacherStudent, setup_logging
from data import UnlabeledDataset
from segment_anything.Teacher_model import Teacher_model
from segment_anything.build_teacher import teacher_model_registry
from argparse import ArgumentParser
import os

parser = ArgumentParser(
    description=("Runs knowledge distillation between SAM.image_encoder(default ViT-H) and a ViT-B")
)
def get_args_parser():
    parser.add_argument("--input", type=str, required=True, help="Path to a folder of images")
    parser.add_argument("--output", type=str, required=True, help="Path to the directory where ViT-B.ckpt will be saved")
    parser.add_argument("--model-type", type=str, required=True, help="The type of teacher model to load")
    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the SAM checkpoint to use for KD")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run KD on")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    #
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    #
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument("--grad-accumulate-steps", type=int, default=1, help="Number of gradient accumulation steps.")


    return parser

def main(args):

    logger = setup_logging(args.output)

    # Data Loading
    print("Loading dataset")
    dataset = UnlabeledDataset(image_folder=args.input)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Models Initialization
    print("Loading teacher model...")
    teacher = teacher_model_registry[args.model_type](checkpoint=args.checkpoint) # ViT-H embed_dim=1280
    teacher = Teacher_model(teacher)
    print("Creating student model...")
    student = StudentImageEncoderViT() # ViT-B embed_dim=768

    unified_model = UnifiedTeacherStudent(teacher_model=teacher, student_model=student).to(args.device)

    criterion = DistillationMSENormalizedLoss(student_feature_dim=student.embed_dim, teacher_feature_dim=teacher.embed_dim)
    optimizer = create_optimizer(args, unified_model.student_model.parameters())

    # Scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Training Loop
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        unified_model.student_model.train()
        unified_model.teacher_model.eval()
        for batch_idx, images in enumerate(dataloader):
            images = [img.to(args.device) for img in images]

            teacher_features, student_output = unified_model(images)
            loss = criterion(student_output, teacher_features)

            loss.backward()
            # Only update every grad_accumulate_steps
            if (batch_idx + 1) % args.grad_accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            # Print status
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
                
        if (batch_idx + 1) % args.grad_accumulate_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= len(dataloader)
        logger.info(f'Epoch: {epoch}, Loss: {epoch_loss}, LR: {optimizer.param_groups[0]["lr"]}')

        lr_scheduler.step()

        # Save model after each epoch (change the path and filename as needed)
        if (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
            model_path = os.path.join(args.output, f"student_model_epoch_{epoch}.pth")
            torch.save(student.state_dict(), model_path)
            logger.info(f'Model saved to {model_path}')

    print("Training Finished!")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)