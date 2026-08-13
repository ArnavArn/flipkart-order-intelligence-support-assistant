"""Interactive / one-shot CLI for the Part 3 support agent.

Usage:
    python -m part3_agent.run_agent --thread-id conv-A
    python -m part3_agent.run_agent --thread-id conv-A --message "What is the return window for apparel?"

With no --message, drops into an interactive loop (type "exit" to quit). Every turn on the same
--thread-id shares state (last_order_id, last_order_features, last_image_path); a new
--thread-id starts with none of that state.
"""
from __future__ import annotations

import argparse
import json

from part3_agent import config
from part3_agent.graph import build_graph, invoke_with_trace


def run_one_turn(graph, user_input: str, thread_id: str) -> None:
    state, path = invoke_with_trace(graph, user_input, thread_id)

    print(f"\nUSER: {user_input}")
    if state.get("injection_blocked"):
        print(f"[guard_input] BLOCKED -- matched pattern: {state.get('injection_pattern')}")
    else:
        print("[guard_input] clean")
        print(f"[classify_intent] {state.get('matched_fewshot')}")
        if state.get("intent") in ("policy", "unknown"):
            print(f"[retrieve] top_score={state.get('top_score', 0.0):.4f}")
        else:
            print(f"[call_tool] {state.get('tool_name')} -> {json.dumps(state.get('tool_result'))}")
    print(f"[verify_output] grounded={state.get('grounded')}")
    print(f"Node path: {' -> '.join(path)}")
    print("AGENT:", json.dumps(state.get("final"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 3 Flipkart support agent CLI")
    parser.add_argument("--thread-id", default="cli-session", help="Conversation thread id")
    parser.add_argument("--message", default=None, help="Single message; skips the REPL loop")
    args = parser.parse_args()

    mode = "MOCK_LLM" if config.MOCK_LLM else "LIVE_LLM"
    print(f"Mode: {mode} (USE_LIVE_LLM={'1' if not config.MOCK_LLM else 'unset'}) "
          f"thread_id={args.thread_id}")

    graph = build_graph()

    if args.message:
        run_one_turn(graph, args.message, args.thread_id)
        return

    print("Type a message, or 'exit' to quit.")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        run_one_turn(graph, user_input, args.thread_id)


if __name__ == "__main__":
    main()
