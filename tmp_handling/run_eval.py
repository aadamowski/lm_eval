#!/usr/bin/env python3
import os
import sys
import argparse
import datetime

# Ensure current directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import metrics to ensure they are registered in the current process
try:
    import custom_eval.metrics
    print("Successfully imported custom_eval.metrics")
except ImportError as e:
    print(f"Failed to import custom_eval.metrics: {e}")
    sys.exit(1)

from lm_eval._cli.run import Run

def run_eval():
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = "dummy"
    
    # Command line args we are simulating
    # We use sys.argv for argparse to consume
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(f"eval_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    sys.argv = [
        "run_eval.py",
        "run",
        "--model", "openai-chat-completions",
        "--model_args", "model=tested_model,base_url=http://127.0.0.1:8080/v1/chat/completions,api_key=dummy,tokenizer=gpt2",
        "--tasks", "custom_eval/tmp_handling_eval.yaml",
        "--include_path", "custom_eval/",
        "--limit", "50",
        "--apply_chat_template",
        "--output_path", output_dir,
        "--log_samples",
    ]
    print(f"Outputs will be saved to: {output_dir}")
    
    # Use argparse to mimic the CLI parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_subcommand = Run(subparsers)
    
    args = parser.parse_args()
    
    if args.command != "run":
        print(f"Expected subcommand 'run', got {args.command}")
        sys.exit(1)
        
    # Call the static _execute method
    Run._execute(args)

if __name__ == "__main__":
    run_eval()
