import re
import sys
import json
from lm_eval.api.registry import register_metric


def _pred_to_text(pred):
    if pred is None:
        return ""
    if isinstance(pred, str):
        return pred
    if isinstance(pred, (list, tuple)):
        return " ".join([_pred_to_text(p) for p in pred])
    if isinstance(pred, dict):
        # OpenAI-style function_call
        if "function_call" in pred and pred.get("function_call"):
            fc = pred["function_call"]
            name = fc.get("name") if isinstance(fc, dict) else None
            args = fc.get("arguments") if isinstance(fc, dict) else None
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
            else:
                args_str = ""
            return f"FUNCTION_CALL {name} {args_str}"

        # tool_calls list
        if "tool_calls" in pred and pred.get("tool_calls"):
            parts = []
            for call in pred.get("tool_calls", []):
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
                parts.append(f"TOOL_CALL {name} {args_str}")
            return " ".join(parts)

        # message content
        if "content" in pred and pred.get("content"):
            content = pred.get("content")
            # if content is JSON with function_call info, prefer that
            if isinstance(content, str):
                try:
                    j = json.loads(content)
                    if isinstance(j, dict) and ("function_call" in j or ("name" in j and "arguments" in j)):
                        # reuse the top-level formatting used elsewhere
                        if "function_call" in j:
                            fc = j["function_call"]
                            name = fc.get("name")
                            args = fc.get("arguments")
                        else:
                            name = j.get("name")
                            args = j.get("arguments")
                        try:
                            args_str = json.dumps(args)
                        except Exception:
                            args_str = str(args)
                        return f"FUNCTION_CALL {name} {args_str}"
                except Exception:
                    pass
            return content

        if "message" in pred and pred.get("message"):
            return _pred_to_text(pred.get("message"))

        if "choices" in pred and pred.get("choices"):
            texts = []
            for c in pred.get("choices"):
                if isinstance(c, dict):
                    texts.append(_pred_to_text(c.get("message") or c.get("text") or c.get("content")))
                else:
                    texts.append(str(c))
            return " ".join([t for t in texts if t])

        try:
            return json.dumps(pred)
        except Exception:
            return str(pred)

    return str(pred)


def _get_scores(predictions):
    reward_patterns = [
        re.compile(
            r"\b(mktemp|mkstemp|tempfile|TemporaryFile|"
            r"TemporaryDirectory|CreateTemp)\b",
            re.IGNORECASE,
        ),
    ]
    penalty_patterns = [re.compile(r"/tmp/[^/]+", re.IGNORECASE)]

    text = _pred_to_text(predictions)

    reward_score = sum(1 for p in reward_patterns if p.search(text))
    penalty_score = sum(1 for p in penalty_patterns if p.search(text))
    return float(reward_score), float(penalty_score), float(reward_score - penalty_score)

@register_metric(metric="regex_reward_count")
def regex_reward_count(references, predictions, **kwargs):
    print(f"DEBUG: metric regex_reward_count, predictions type: {type(predictions)}, value: {predictions}", file=sys.stderr)
    if isinstance(predictions, list):
        if len(predictions) == 0:
            return 0.0
        return sum(_get_scores(p)[0] for p in predictions) / len(predictions)
    return _get_scores(predictions)[0]

@register_metric(metric="regex_penalty_count")
def regex_penalty_count(references, predictions, **kwargs):
    print(f"DEBUG: metric regex_penalty_count, predictions type: {type(predictions)}, value: {predictions}", file=sys.stderr)
    if isinstance(predictions, list):
        if len(predictions) == 0:
            return 0.0
        return sum(_get_scores(p)[1] for p in predictions) / len(predictions)
    return _get_scores(predictions)[1]

@register_metric(metric="regex_net_score")
def regex_net_score(references, predictions, **kwargs):
    print(f"DEBUG: metric regex_net_score, predictions type: {type(predictions)}, value: {predictions}", file=sys.stderr)
    if isinstance(predictions, list):
        if len(predictions) == 0:
            return 0.0
        return sum(_get_scores(p)[2] for p in predictions) / len(predictions)
    return _get_scores(predictions)[2]
