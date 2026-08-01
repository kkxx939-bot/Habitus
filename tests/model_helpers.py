"""ModelClient 测试替身共享的最小请求准备器。"""

from ModelClient import ChatRequest, PreparedChatRequest


def prepare_chat_request(request: ChatRequest, *, stream: bool) -> PreparedChatRequest:
    """为不测试协议序列化的 Provider 替身创建稳定的已准备请求。"""

    return PreparedChatRequest(
        request=request,
        body=b"{}",
        model_visible_body=b"{}",
        reserved_output_tokens=request.max_output_tokens or 0,
        stream=stream,
    )


__all__ = ["prepare_chat_request"]
