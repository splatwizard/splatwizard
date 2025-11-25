import torch
from loguru import logger

from splatwizard.utils.misc import wrap_str


class SummeryWriterWrapper:
    def __init__(self, tb_writer, prefix=''):
        self.tb_writer = tb_writer
        self.prefix = prefix


    def add_scalar(self,
                   tag,
                   scalar_value,
                   global_step=None,
                   walltime=None,
                   new_style=False,
                   double_precision=False,
                   ):
        self.tb_writer.add_scalar(
            self.prefix + '/' + tag,
            scalar_value,
            global_step,
            walltime,
            new_style,
            double_precision,
        )

    def add_histogram(
        self,
        tag,
        values,
        global_step=None,
        bins="tensorflow",
        walltime=None,
        max_bins=None,
    ):
        self.tb_writer.add_histogram(
            self.prefix + '/' + tag,
            values,
            global_step,
            bins,
            walltime,
            max_bins,
        )



def format_results_loss(logs: dict, col_name: bool = False) -> str:
    """Format a dictionary of logs as either a one-row string (if col_name
    is False) or a two-row string (if col_name is True).

    Args:
        logs (dict): output of the loss function containing different metrics.

    Returns:
        str: A one or two-row strings.
    """
    # Log first col name if needed then the values.
    msg = ''
    if col_name:
        for k in logs:
            msg += f'{k:<15}'
        msg += '\n'

    for _, v in logs.items():
        if isinstance(v, list):
            continue
        v = v.item() if isinstance(v, torch.Tensor) else v
        v = f'{v}' if isinstance(v, int) else f'{v:5.3f}'
        msg += f'{v:<15}'

    return msg


def setup_tensorboard(output_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_FOUND = True
        logger.info("found tf board")
    except ImportError:
        TENSORBOARD_FOUND = False
        SummaryWriter = None
        logger.info("not found tf board")


    # if not pipe.model_path:
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str=os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     pipe.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    logger.info(wrap_str("Tensorboard output folder:", output_dir))
    # os.makedirs(pipe.model_path, exist_ok = True)
    # with open(os.path.join(pipe.model_path, "cfg_args"), 'w') as cfg_log_f:
    #     cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        assert SummaryWriter
        tb_writer = SummeryWriterWrapper(SummaryWriter(output_dir))
    else:
        logger.info("Tensorboard not available: not logging progress")
    return tb_writer


def log_training_results(
        tb_writer,
        iteration,
        psnr_val,
        bpp,
        l2_loss,
        loss,
        z_loss,
        weight_sim,
        temperature,
        lr,
        aux_loss
        # elapsed
):
    order = 0
    tb_writer.add_scalar(f'{order}/train/PSNR', psnr_val.item(), iteration)
    tb_writer.add_scalar(f'{order}/train/l2_loss', l2_loss.item(), iteration)
    tb_writer.add_scalar(f'{order}/train/bpp', bpp.item(), iteration)
    tb_writer.add_scalar(f'{order}/train/total_loss', loss.item(), iteration)
    tb_writer.add_scalar(f'{order}/train/psnr_bpp', psnr_val.item() / bpp.item(), iteration)
    # tb_writer.add_scalar(f'{order}/train/z_loss', z_loss.item(), iteration)
    # tb_writer.add_scalar(f'{order}/train/weight_abs_cosine', weight_sim.item(), iteration)
    tb_writer.add_scalar(f'{order}/train/temperature', temperature, iteration)
    tb_writer.add_scalar(f'{order}/train/lr', lr, iteration)
    tb_writer.add_scalar(f'{order}/train/aux_loss', aux_loss.item(), iteration)


    # tb_writer.add_scalar(f'{order}/iter_time', elapsed, iteration)


def log_valid_results(
        tb_writer,
        iteration,
        psnr_val,
        psnr_val_ref,
        bpp,

):
    order = 1
    tb_writer.add_scalar(f'{order}/valid/PSNR', psnr_val, iteration)
    tb_writer.add_scalar(f'{order}/valid/PSNR_ref', psnr_val_ref, iteration)
    tb_writer.add_scalar(f'{order}/valid/bpp', bpp, iteration)
    tb_writer.add_scalar(f'{order}/valid/psnr_bpp', psnr_val / bpp, iteration)