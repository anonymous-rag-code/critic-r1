import argparse

from verl.model_merger.base_model_merger import BaseModelMerger


class Config:
    def __init__(self, target_dir: str, trust_remote_code: bool = False):
        self.target_dir = target_dir
        self.trust_remote_code = trust_remote_code


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and save LoRA adapters from a VERL FSDP checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the VERL checkpoint directory, e.g., /path/to/checkpoint/global_step_xxx.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path or HuggingFace name of the base model, e.g., Qwen/Qwen2.5-3B-Instruct.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./extracted_lora",
        help="Directory to save the extracted LoRA adapter.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading the model configuration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        target_dir=args.target_dir,
        trust_remote_code=args.trust_remote_code,
    )

    merger = BaseModelMerger(
        config=config,
        checkpoint_path=args.checkpoint_path,
        hf_model_config_path=args.base_model_path,
    )

    merger.extract_and_save()


if __name__ == "__main__":
    main()