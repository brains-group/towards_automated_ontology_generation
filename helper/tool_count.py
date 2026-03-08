def tool_call_count(ai_msg) -> int:
    """
    LangChain's AIMessage stores tool calls either on `tool_calls`
    or under OpenAI-compatible `additional_kwargs["tool_calls"]`.
    """
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return len(ai_msg.tool_calls)
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    if isinstance(ak, dict) and isinstance(ak.get("tool_calls"), list):
        return len(ak["tool_calls"])
    return 0