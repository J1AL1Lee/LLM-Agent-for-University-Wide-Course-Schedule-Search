from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jiaowu_rag.config import Settings  # noqa: E402
from jiaowu_rag.deepseek import DeepSeekToolCallingAssistant  # noqa: E402
from jiaowu_rag.retriever import ChromaScheduleRetriever  # noqa: E402
from jiaowu_rag.tools import ScheduleToolbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-turn BJUT schedule agent with SQL/Chroma tool routing."
    )
    parser.add_argument("--top-k", type=int, default=None, help="Maximum evidence rows per tool call.")
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print each tool name, result count, and error after an answer.",
    )
    return parser.parse_args()


async def run_chat(args: argparse.Namespace) -> int:
    settings = Settings.from_env(PROJECT_ROOT)
    if not settings.deepseek_api_key:
        print("错误：未配置 DEEPSEEK_API_KEY。请先在项目根目录的 .env 中设置。", file=sys.stderr)
        return 2
    top_k = args.top_k or settings.default_top_k
    if top_k < 1 or top_k > settings.max_top_k:
        print(
            f"错误：--top-k 必须在 1 到 {settings.max_top_k} 之间。",
            file=sys.stderr,
        )
        return 2

    chroma_dir = Path(settings.chroma_dir)
    if not chroma_dir.is_absolute():
        chroma_dir = PROJECT_ROOT / chroma_dir
    retriever = ChromaScheduleRetriever(
        PROJECT_ROOT,
        persist_dir=chroma_dir,
        collection_name=settings.chroma_collection,
    )
    toolbox = ScheduleToolbox(retriever, max_results=settings.max_top_k)
    assistant = DeepSeekToolCallingAssistant(settings)
    history: list[dict[str, str]] = []

    print("教务 Agent 已启动。输入 quit 或 exit 退出，Ctrl+C 也可结束。")
    print(
        f"Chroma collection={retriever.collection_name}，records={retriever.count}，"
        f"model={assistant.model_name}"
    )
    try:
        while True:
            try:
                user_input = (await asyncio.to_thread(input, "\n你: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break
            if user_input.lower() in {"quit", "exit"}:
                print("已退出。")
                break
            if not user_input:
                continue

            try:
                outcome = await assistant.run(
                    user_input,
                    toolbox,
                    result_limit=top_k,
                    conversation_history=history,
                )
            except Exception as exc:
                print(f"Agent 调用失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            print(f"\nAgent: {outcome.answer}")
            if args.show_tools:
                for index, call in enumerate(outcome.calls, start=1):
                    suffix = f" error={call.error}" if call.error else ""
                    print(
                        f"  tool[{index}] {call.name} results={call.result_count}{suffix}"
                    )

            history.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": outcome.answer},
                ]
            )
            if settings.max_chat_history_messages:
                history = history[-settings.max_chat_history_messages :]
            else:
                history.clear()
    finally:
        await assistant.aclose()
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    raise SystemExit(asyncio.run(run_chat(args)))


if __name__ == "__main__":
    main()
