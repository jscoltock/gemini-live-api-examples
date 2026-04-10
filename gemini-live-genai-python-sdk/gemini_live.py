import asyncio
import inspect
import logging
import traceback

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types

class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """
    def __init__(self, api_key, model, input_sample_rate, tools=None, tool_mapping=None):
        """
        Initializes the GeminiLive client.

        Args:
            api_key (str): The Gemini API Key.
            model (str): The model name to use.
            input_sample_rate (int): The sample rate for audio input.
            tools (list, optional): List of tools to enable. Defaults to None.
            tool_mapping (dict, optional): Mapping of tool names to functions. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None, notification_queue=None, file_input_queue=None):
        import agent_config
        gemini_cfg = agent_config.get_gemini_session()
        system_prompt = gemini_cfg.get("system_prompt", "")
        voice_name = gemini_cfg.get("voice", "Puck")

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=system_prompt)]) if system_prompt else None,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            tools=self.tools,
        )
        
        logger.info(f"Connecting to Gemini Live with model={self.model}")
        try:
          async with self.client.aio.live.connect(model=self.model, config=config) as session:
            logger.info("Gemini Live session opened successfully")
            
            async def send_audio():
                try:
                    while True:
                        chunk = await audio_input_queue.get()
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_audio task cancelled")
                except Exception as e:
                    logger.error(f"send_audio error: {e}\n{traceback.format_exc()}")

            async def send_video():
                try:
                    while True:
                        chunk = await video_input_queue.get()
                        logger.info(f"Sending video frame to Gemini: {len(chunk)} bytes")
                        await session.send_realtime_input(
                            video=types.Blob(data=chunk, mime_type="image/jpeg")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_video task cancelled")
                except Exception as e:
                    logger.error(f"send_video error: {e}\n{traceback.format_exc()}")

            async def send_text():
                try:
                    while True:
                        text = await text_input_queue.get()
                        logger.info(f"Sending text to Gemini: {text}")
                        await session.send_realtime_input(text=text)
                except asyncio.CancelledError:
                    logger.debug("send_text task cancelled")
                except Exception as e:
                    logger.error(f"send_text error: {e}\n{traceback.format_exc()}")

            async def send_notifications():
                """Read from notification_queue and inject messages via realtime input."""
                try:
                    while True:
                        msg = await notification_queue.get()
                        if msg is None:
                            break
                        logger.info(f"Sending notification to Gemini: {msg}")
                        await session.send_realtime_input(text=msg)
                except asyncio.CancelledError:
                    logger.debug("send_notifications task cancelled")
                except Exception as e:
                    logger.error(f"send_notifications error: {e}\n{traceback.format_exc()}")

            # Text-based extensions we can embed directly
            _TEXT_EXTS = {".txt", ".csv", ".json", ".xml", ".html", ".htm", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".bash", ".zsh", ".log", ".sql", ".rst", ".tex"}

            async def send_files():
                """Read files from file_input_queue and send to Gemini."""
                import os as _os
                import tempfile
                try:
                    while True:
                        file_msg = await file_input_queue.get()
                        if file_msg is None:
                            break
                        data = file_msg["data"]
                        mime = file_msg["mime_type"]
                        fname = file_msg["file_name"]
                        ext = _os.path.splitext(fname)[1].lower()
                        size_kb = len(data) / 1024
                        logger.info(f"Sending file to Gemini: {fname} ({size_kb:.1f} KB, {mime})")
                        try:
                            if mime.startswith("image/"):
                                # Images: send as realtime video frame (same path as camera)
                                await session.send_realtime_input(
                                    video=types.Blob(data=data, mime_type=mime)
                                )
                                await session.send_realtime_input(
                                    text=f'The user uploaded an image named "{fname}". Please analyze it.'
                                )
                            elif mime.startswith("text/") or ext in _TEXT_EXTS:
                                # Text files: embed content directly in text message
                                try:
                                    content = data.decode("utf-8")
                                except UnicodeDecodeError:
                                    content = data.decode("latin-1")
                                # Truncate very large files
                                if len(content) > 50000:
                                    content = content[:50000] + f"\n\n... (truncated, {size_kb:.0f} KB total)"
                                await session.send_realtime_input(
                                    text=f'The user uploaded a text file named "{fname}":\n\n{content}'
                                )
                            else:
                                # PDFs and other binary: upload via Files API, then reference
                                suffix = ext if ext else ".bin"
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    tmp.write(data)
                                    tmp_path = tmp.name
                                try:
                                    uploaded = self.client.files.upload(file=tmp_path)
                                    logger.info(f"Uploaded {fname} to Files API: {uploaded.name}")
                                    # Poll until active (max 60s)
                                    for _ in range(60):
                                        if uploaded.state and uploaded.state.name != "PROCESSING":
                                            break
                                        await asyncio.sleep(1)
                                        uploaded = self.client.files.get(name=uploaded.name)
                                    if uploaded.state and uploaded.state.name == "ACTIVE":
                                        await session.send_client_content(
                                            turns=types.Content(
                                                role="user",
                                                parts=[
                                                    types.Part.from_uri(
                                                        file_uri=uploaded.uri,
                                                        mime_type=mime,
                                                    ),
                                                    types.Part(text=f'The user uploaded a file named "{fname}". Please analyze it.'),
                                                ]
                                            )
                                        )
                                    else:
                                        await event_queue.put({"type": "gemini", "text": f"[File {fname} failed to process: {uploaded.state}]"})
                                finally:
                                    _os.unlink(tmp_path)
                                    try:
                                        self.client.files.delete(name=uploaded.name)
                                    except Exception:
                                        pass
                            logger.info(f"File {fname} sent to Gemini successfully")
                        except Exception as e:
                            logger.error(f"Error sending file {fname}: {e}\n{traceback.format_exc()}")
                            await event_queue.put({"type": "gemini", "text": f"[Error sending file {fname}: {e}]"})
                except asyncio.CancelledError:
                    logger.debug("send_files task cancelled")
                except Exception as e:
                    logger.error(f"send_files error: {e}\n{traceback.format_exc()}")

            event_queue = asyncio.Queue()

            async def _handle_tool_call(tool_call):
                """Process tool calls in a separate task so the receive loop keeps draining."""
                function_responses = []
                for fc in tool_call.function_calls:
                    func_name = fc.name
                    args = fc.args or {}
                    
                    if func_name in self.tool_mapping:
                        try:
                            tool_func = self.tool_mapping[func_name]
                            if inspect.iscoroutinefunction(tool_func):
                                result = await tool_func(**args)
                            else:
                                loop = asyncio.get_running_loop()
                                result = await loop.run_in_executor(None, lambda: tool_func(**args))
                        except Exception as e:
                            result = f"Error: {e}"
                        
                        function_responses.append(types.FunctionResponse(
                            name=func_name,
                            id=fc.id,
                            response={"result": result}
                        ))
                        await event_queue.put({"type": "tool_call", "name": func_name, "args": args, "result": result})
                
                if function_responses:
                    await session.send_tool_response(function_responses=function_responses)

            async def receive_loop():
                try:
                    while True:
                        async for response in session.receive():
                            logger.debug(f"Received response from Gemini: {response}")
                            
                            # Log the raw response type for debugging
                            if response.go_away:
                                logger.warning(f"Received GoAway from Gemini: {response.go_away}")
                            if response.session_resumption_update:
                                logger.info(f"Session resumption update: {response.session_resumption_update}")
                            
                            # Capture usage metadata
                            usage = response.usage_metadata
                            
                            server_content = response.server_content
                            tool_call = response.tool_call
                            
                            if usage:
                                usage_data = {}
                                if usage.prompt_token_count is not None:
                                    usage_data["prompt_token_count"] = usage.prompt_token_count
                                if usage.response_token_count is not None:
                                    usage_data["response_token_count"] = usage.response_token_count
                                if usage.total_token_count is not None:
                                    usage_data["total_token_count"] = usage.total_token_count
                                if usage.cached_content_token_count is not None:
                                    usage_data["cached_content_token_count"] = usage.cached_content_token_count
                                if usage.thoughts_token_count is not None:
                                    usage_data["thoughts_token_count"] = usage.thoughts_token_count
                                # Include modality breakdown if available
                                if usage.prompt_tokens_details:
                                    usage_data["prompt_tokens_details"] = [
                                        {"modality": d.modality, "token_count": d.token_count}
                                        for d in usage.prompt_tokens_details
                                        if d.token_count
                                    ]
                                if usage.response_tokens_details:
                                    usage_data["response_tokens_details"] = [
                                        {"modality": d.modality, "token_count": d.token_count}
                                        for d in usage.response_tokens_details
                                        if d.token_count
                                    ]
                                if usage_data:
                                    logger.info(f"Usage metadata: {usage_data}")
                                    await event_queue.put({"type": "usage", "usage": usage_data})
                            
                            if server_content:
                                if server_content.model_turn:
                                    for part in server_content.model_turn.parts:
                                        if part.inline_data:
                                            if inspect.iscoroutinefunction(audio_output_callback):
                                                await audio_output_callback(part.inline_data.data)
                                            else:
                                                audio_output_callback(part.inline_data.data)
                                
                                if server_content.input_transcription and server_content.input_transcription.text:
                                    await event_queue.put({"type": "user", "text": server_content.input_transcription.text})
                                
                                if server_content.output_transcription and server_content.output_transcription.text:
                                    await event_queue.put({"type": "gemini", "text": server_content.output_transcription.text})
                                
                                if server_content.turn_complete:
                                    await event_queue.put({"type": "turn_complete"})
                                
                                if server_content.interrupted:
                                    if audio_interrupt_callback:
                                        if inspect.iscoroutinefunction(audio_interrupt_callback):
                                            await audio_interrupt_callback()
                                        else:
                                            audio_interrupt_callback()
                                    await event_queue.put({"type": "interrupted"})

                            if tool_call:
                                # Fire and forget — lets receive loop keep draining
                                asyncio.create_task(_handle_tool_call(tool_call))
                        
                        # session.receive() iterator ended (e.g. after turn_complete) — re-enter to keep listening
                        logger.debug("Gemini receive iterator completed, re-entering receive loop")

                except asyncio.CancelledError:
                    logger.debug("receive_loop task cancelled")
                except Exception as e:
                    logger.error(f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                    await event_queue.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                finally:
                    logger.info("receive_loop exiting")
                    await event_queue.put(None)

            send_audio_task = asyncio.create_task(send_audio())
            send_video_task = asyncio.create_task(send_video())
            send_text_task = asyncio.create_task(send_text())
            notification_task = asyncio.create_task(send_notifications()) if notification_queue else None
            file_task = asyncio.create_task(send_files()) if file_input_queue else None
            receive_task = asyncio.create_task(receive_loop())

            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    if isinstance(event, dict) and event.get("type") == "error":
                        # Just yield the error event, don't raise to keep the stream alive if possible or let caller handle
                        yield event
                        break 
                    yield event
            finally:
                logger.info("Cleaning up Gemini Live session tasks")
                send_audio_task.cancel()
                send_video_task.cancel()
                send_text_task.cancel()
                if notification_task:
                    notification_task.cancel()
                if file_task:
                    file_task.cancel()
                receive_task.cancel()
        except Exception as e:
            logger.error(f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Gemini Live session closed")
