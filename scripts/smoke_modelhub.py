import argparse
import json

from dotenv import load_dotenv

from src.models import ModelHub


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the configured ModelHub API.")
    parser.add_argument("--prompt", default="Reply with exactly: MODELHUB_OK")
    parser.add_argument("--image", help="Optional local image path or public image URL")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    load_dotenv()
    prompt = (
        {"text": args.prompt, "images": [args.image]}
        if args.image
        else args.prompt
    )

    with ModelHub() as model:
        result = model.generate_with_metadata(prompt, max_out_len=args.max_tokens)

    print(result["content"])
    print(
        json.dumps(
            {
                "model": result["model"],
                "usage": result["usage"],
                "finish_reason": result["finish_reason"],
                "log_id": result["log_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
