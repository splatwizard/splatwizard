import argparse
import sys


def train_handler(args):
    from splatwizard.scripts.train import main as cmd_train
    cmd_train(sys.argv[2:], prog='train')


def eval_handler(args):
    from splatwizard.scripts.eval import main as cmd_eval
    cmd_eval(sys.argv[2:], prog='eval')


def recon_handler(args):
    from splatwizard.scripts.reconstruct import main as cmd_recon
    cmd_recon(sys.argv[2:], prog='recon')


def main():
    parser = argparse.ArgumentParser(description="CLI tool for Splatwizard")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    parser_cmd1 = subparsers.add_parser("train", help="Train model", add_help=False)
    parser_cmd1.set_defaults(func=train_handler)

    parser_cmd2 = subparsers.add_parser("eval", help="Evaluate model", add_help=False)
    parser_cmd2.set_defaults(func=eval_handler)

    parser_cmd3 = subparsers.add_parser("recon", help="Reconstruct mesh from trained model", add_help=False)
    parser_cmd3.set_defaults(func=eval_handler)

    known_args, extra_args = parser.parse_known_args()

    if hasattr(known_args, 'func'):
        known_args.func(known_args)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()