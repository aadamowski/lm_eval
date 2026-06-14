import re
import json
from lm_eval.api.task import Task


class FreeformRegexEval(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_patterns = [
            re.compile(
                r"\b(mktemp|mkstemp|tempfile|TemporaryFile|"
                r"TemporaryDirectory|CreateTemp)\b",
                re.IGNORECASE,
            ),
        ]
        self.penalty_patterns = [re.compile(r"/tmp/[^/]+", re.IGNORECASE)]

    @property
    def dataset_path(self):
        return "json"

    @property
    def dataset_kwargs(self):
        return {"data_files": {"test": "data/prompts.jsonl"}}

    def process_results(self, doc, results):
        def _extract_text(obj):
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, (list, tuple)):
                parts = [_extract_text(x) for x in obj]
                return " ".join([p for p in parts if p])
            if isinstance(obj, dict):
                # OpenAI-style function_call
                if "function_call" in obj and obj.get("function_call"):
                    fc = obj["function_call"]
                    name = fc.get("name") if isinstance(fc, dict) else None
                    args = fc.get("arguments") if isinstance(fc, dict) else None
                    args_str = ""
                    if isinstance(args, str):
                        try:
                            args_parsed = json.loads(args)
                            args_str = json.dumps(args_parsed)
                        except Exception:
                            args_str = args
                    elif args is not None:
                        try:
                            args_str = json.dumps(args)
                        except Exception:
                            args_str = str(args)
                    return f"FUNCTION_CALL {name} {args_str}"

                # tool_calls list
                if "tool_calls" in obj and obj.get("tool_calls"):
                    calls = []
                    for call in obj.get("tool_calls", []):
                        name = call.get("name")
                        args = call.get("arguments")
                        if isinstance(args, str):
                            try:
                                args_parsed = json.loads(args)
                                args_str = json.dumps(args_parsed)
                            except Exception:
                                args_str = args
                        else:
                            try:
                                args_str = json.dumps(args)
                            except Exception:
                                args_str = str(args)
                        calls.append(f"TOOL_CALL {name} {args_str}")
                    return " ".join(calls)

                # message-like content
                if "content" in obj and obj.get("content"):
                    content = obj.get("content")
                    # If assistant `content` is a JSON-encoded OpenAI-style function_call
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                # already-wrapped function_call
                                if "function_call" in parsed and isinstance(parsed.get("function_call"), dict):
                                    fc = parsed.get("function_call")
                                    name = fc.get("name")
                                    args = fc.get("arguments")
                                    args_str = ""
                                    if isinstance(args, str):
                                        try:
                                            args_parsed = json.loads(args)
                                            args_str = json.dumps(args_parsed)
                                        except Exception:
                                            args_str = args
                                    elif args is not None:
                                        try:
                                            args_str = json.dumps(args)
                                        except Exception:
                                            args_str = str(args)
                                    return f"FUNCTION_CALL {name} {args_str}"

                                # direct function_call-like JSON: {"name":..., "arguments": {...}}
                                if "name" in parsed and "arguments" in parsed:
                                    name = parsed.get("name")
                                    args = parsed.get("arguments")
                                    args_str = ""
                                    if isinstance(args, str):
                                        try:
                                            args_parsed = json.loads(args)
                                            args_str = json.dumps(args_parsed)
                                        except Exception:
                                            args_str = args
                                    elif args is not None:
                                        try:
                                            args_str = json.dumps(args)
                                        except Exception:
                                            args_str = str(args)
                                    return f"FUNCTION_CALL {name} {args_str}"
                        except Exception:
                            pass
                    return content

                if "message" in obj and obj.get("message"):
                    return _extract_text(obj.get("message"))

                if "choices" in obj and obj.get("choices"):
                    texts = []
                    for c in obj.get("choices"):
                        if isinstance(c, dict):
                            # choice may contain message or text
                            texts.append(_extract_text(c.get("message") or c.get("text") or c.get("content")))
                        else:
                            texts.append(str(c))
                    return " ".join([t for t in texts if t])

                try:
                    return json.dumps(obj)
                except Exception:
                    return str(obj)

            return str(obj)

        generated_text = _extract_text(results)

        reward_score = sum(1 for p in self.reward_patterns if p.search(generated_text))
        penalty_score = sum(1 for p in self.penalty_patterns if p.search(generated_text))
        final_score = float(reward_score - penalty_score)

        result = {
            "regex_reward_count": float(reward_score),
            "regex_penalty_count": float(penalty_score),
            "regex_net_score": final_score,
        }
        
        return result

    def aggregation(self):
        return {
            "regex_reward_count": "mean",
            "regex_penalty_count": "mean",
            "regex_net_score": "mean",
        }

    def higher_is_better(self):
        return {
            "regex_reward_count": True,
            "regex_penalty_count": False,
            "regex_net_score": True,
        }

    def doc_to_text(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        return ""
