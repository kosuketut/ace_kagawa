# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

"""Tokkio RAG service with filler phrase support."""

import asyncio
import json
import random
import time

from loguru import logger
from nvidia_pipecat.frames.nvidia_rag import NvidiaRAGCitation, NvidiaRAGCitationsFrame
from nvidia_pipecat.services.nvidia_rag import NvidiaRAGService
from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced
from openai.types.chat import ChatCompletionMessageParam
from pipecat.frames.frames import ErrorFrame, TextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


@traceable
class TokkioNvidiaRAGService(NvidiaRAGService):
    def __init__(
        self,
        collection_name: str,
        filler: list[str],
        time_delay: float = 6.0,
        rag_server_url: str = "http://localhost:8081/v1",
        stop_words: list | None = None,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 128,
        use_knowledge_base: bool = True,
        vdb_top_k: int = 12,
        reranker_top_k: int = 5,
        multimodal_reranker_top_k: int = 10,
        enable_reranker: bool = True,
        enable_citations: bool = True,
        suffix_prompt: str | None = None,
        **kwargs,
    ):
        self.filler = filler
        self.time_delay = time_delay
        self.timeout = 120
        self.vdb_top_k = vdb_top_k
        self.reranker_top_k = reranker_top_k
        self.multimodal_reranker_top_k = multimodal_reranker_top_k
        self.enable_reranker = enable_reranker

        super().__init__(
            collection_name=collection_name,
            rag_server_url=rag_server_url.rstrip("/"),
            stop_words=stop_words,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            use_knowledge_base=use_knowledge_base,
            vdb_top_k=vdb_top_k,
            reranker_top_k=reranker_top_k,
            enable_citations=enable_citations,
            suffix_prompt=suffix_prompt,
            **kwargs,
        )

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="rag")
    async def _get_rag_response(self, request_json: dict):
        rag_url = f"{self.rag_server_url.rstrip('/')}/generate"
        return await self.shared_session.post(rag_url, timeout=self.timeout, json=request_json)

    def _select_reranker_top_k(self, chat_details: list[dict]) -> int:
        multimodal_terms = ("表", "テーブル", "table", "図", "グラフ", "画像", "chart", "figure", "image")
        last_user_message = ""
        for message in reversed(chat_details):
            if message.get("role") == "user":
                last_user_message = str(message.get("content") or "").lower()
                break
        if any(term in last_user_message for term in multimodal_terms):
            return max(self.reranker_top_k, self.multimodal_reranker_top_k)
        return self.reranker_top_k

    def _parse_rag_chunk(self, raw_chunk: str) -> tuple[str, list[NvidiaRAGCitation]]:
        chunk = raw_chunk.strip()
        if not chunk:
            return "", []
        if chunk.startswith("data:"):
            chunk = chunk[len("data:") :].strip()
        if not chunk or chunk == "[DONE]":
            return "", []

        parsed = json.loads(chunk)
        message = ""
        choices = parsed.get("choices") or []
        if choices:
            first_choice = choices[0]
            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                message = delta.get("content") or ""
            response_message = first_choice.get("message")
            if not message and isinstance(response_message, dict):
                message = response_message.get("content") or ""
            if not message:
                message = first_choice.get("content") or ""

        citations = []
        for citation in (parsed.get("citations") or {}).get("results") or []:
            citations.append(
                NvidiaRAGCitation(
                    document_type=str(citation["document_type"]),
                    document_id=str(citation["document_id"]),
                    document_name=str(citation["document_name"]),
                    content=str(citation["content"]).encode(),
                    metadata=str(citation["metadata"]),
                    score=float(citation["score"]),
                )
            )
        return message, citations

    async def _process_context(self, context: OpenAILLMContext):
        try:
            messages: list[ChatCompletionMessageParam] = context.get_messages()
            chat_details = []

            for msg in messages:
                if msg["role"] not in {"system", "user", "assistant"}:
                    raise Exception(f"Unexpected role {msg['role']} found!")
                chat_details.append({"role": msg["role"], "content": msg["content"]})

            if self.suffix_prompt:
                for i in range(len(chat_details) - 1, -1, -1):
                    if chat_details[i]["role"] == "user":
                        chat_details[i]["content"] += f" {self.suffix_prompt}"
                        break

            logger.debug(f"Chat details: {chat_details}")

            if len(chat_details) == 0 or all(msg["content"] == "" for msg in chat_details) or not self.collection_name:
                raise Exception("No query or collection name is provided.")

            reranker_top_k = self._select_reranker_top_k(chat_details)
            vdb_top_k = max(self.vdb_top_k, reranker_top_k)

            request_json = {
                "messages": chat_details,
                "use_knowledge_base": self.use_knowledge_base,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "collection_names": [self.collection_name],
                "collection_name": self.collection_name,
                "vdb_top_k": vdb_top_k,
                "reranker_top_k": reranker_top_k,
                "enable_reranker": self.enable_reranker,
                "stop": self.stop_words or [],
                "enable_citations": self.enable_citations,
            }

            await self.start_ttfb_metrics()

            start_time = time.time()
            first_chunk_received = False
            full_response = ""
            filler_said = False

            async def monitor_request_time():
                nonlocal filler_said
                await asyncio.sleep(self.time_delay)
                if self.filler and not first_chunk_received and not filler_said:
                    filler_said = True
                    await self.push_frame(TTSSpeakFrame(random.choice(self.filler)))

            monitor_task = asyncio.create_task(monitor_request_time())
            resp = await self._get_rag_response(request_json)
            try:
                async for chunk in resp.aiter_lines():
                    if not first_chunk_received:
                        elapsed_time = time.time() - start_time
                        logger.debug(f"Elapsed time: {elapsed_time}")
                        logger.debug(f"Time delay: {self.time_delay}")
                        first_chunk_received = True

                        if not monitor_task.done():
                            monitor_task.cancel()
                            try:
                                await monitor_task
                            except asyncio.CancelledError:
                                pass

                    await self.stop_ttfb_metrics()

                    try:
                        message, citations = self._parse_rag_chunk(chunk)
                    except Exception as exc:
                        logger.debug(f"Parsing RAG response chunk failed. Error: {exc}")
                        continue

                    if not message and not citations:
                        continue
                    full_response += message
                    if citations:
                        scores = [citation.score for citation in citations]
                        types = [citation.document_type for citation in citations]
                        logger.debug(f"Received total {len(citations)} RAG citations")
                        logger.debug(f"Received RAG citation types: {types}")
                        logger.debug(f"Received RAG citation scores: {scores}")
                        await self.push_frame(NvidiaRAGCitationsFrame(citations=citations))
                    if message:
                        await self.push_frame(TextFrame(message))
            finally:
                if not monitor_task.done():
                    monitor_task.cancel()
                await resp.aclose()
            logger.debug(f"Full RAG response: {full_response}")

        except Exception as exc:
            logger.error(f"An error occurred in http request to RAG endpoint, Error: {exc!r}")
            await self.push_error(ErrorFrame("Cannot connect to the RAG endpoint"))
            await self.push_frame(TTSSpeakFrame("RAGサーバーへ接続できません。"))
